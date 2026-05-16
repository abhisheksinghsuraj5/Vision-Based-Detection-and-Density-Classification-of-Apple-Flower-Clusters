import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def calculate_enhanced_flower_pixel_count(image, cluster_polygon):
    """YOUR EXACT Enhanced flower pixel detection - returns COUNTS only"""
    try:
        img_height, img_width = image.shape[:2]
        
        # Create mask for the cluster polygon
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        polygon_int = np.array(cluster_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_int], 255)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # YOUR EXACT 10 detection methods
        white_mask_rgb = np.logical_and.reduce([
            image[:, :, 2] >= 120, image[:, :, 1] >= 120, image[:, :, 0] >= 120
        ])
        
        light_mask = np.logical_and.reduce([
            image[:, :, 2] >= 130, image[:, :, 1] >= 120, image[:, :, 0] >= 100
        ])
        
        hsv_light_mask = np.logical_and(hsv[:, :, 2] >= 130, hsv[:, :, 1] <= 120)
        
        pink_mask = np.logical_and.reduce([
            image[:, :, 2] >= 120, image[:, :, 1] >= 80, image[:, :, 1] <= 220,
            image[:, :, 0] >= 80, image[:, :, 0] <= 200
        ])
        
        lab_light_mask = lab[:, :, 0] >= 140
        
        cream_mask = np.logical_and.reduce([
            image[:, :, 2] >= 140, image[:, :, 1] >= 130, image[:, :, 0] >= 110,
            image[:, :, 2] <= 255, image[:, :, 1] <= 255, image[:, :, 0] <= 220
        ])
        
        lavender_mask = np.logical_and.reduce([
            image[:, :, 2] >= 120, image[:, :, 1] >= 100, image[:, :, 0] >= 130
        ])
        
        brightness_sum = (image[:, :, 2].astype(np.int32) + 
                         image[:, :, 1].astype(np.int32) + 
                         image[:, :, 0].astype(np.int32))
        very_light_mask = np.logical_and(
            np.logical_or.reduce([
                image[:, :, 2] >= 150, image[:, :, 1] >= 150, image[:, :, 0] >= 150
            ]),
            brightness_sum >= 380
        )
        
        hsv_flower_mask = np.logical_and(hsv[:, :, 1] <= 100, hsv[:, :, 2] >= 120)
        
        brightness = ((image[:, :, 0].astype(np.float32) + 
                      image[:, :, 1].astype(np.float32) + 
                      image[:, :, 2].astype(np.float32)) / 3)
        brightness_mask = brightness >= 120
        
        # Combine ALL methods
        flower_mask = np.logical_or.reduce([
            white_mask_rgb, light_mask, hsv_light_mask, pink_mask,
            lab_light_mask, cream_mask, lavender_mask, very_light_mask,
            hsv_flower_mask, brightness_mask
        ])
        
        cluster_flower_mask = np.logical_and(flower_mask, mask > 0)
        flower_pixels_count = np.sum(cluster_flower_mask)
        
        return flower_pixels_count, cluster_flower_mask.astype(np.uint8) * 255
        
    except Exception as e:
        print(f"Error calculating flower pixels: {e}")
        return 0, None



def calculate_edge_pixels_in_flowers(image, flower_mask):
    """Calculate edge pixels inside flower pixels - returns COUNT only"""
    try:
        if flower_mask is None or not np.any(flower_mask):
            return 0, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 250)
        flower_edges = edges & flower_mask
        edge_pixels_count = np.sum(flower_edges > 0)
        
        return edge_pixels_count, flower_edges
        
    except Exception as e:
        print(f"Error calculating edge pixels: {e}")
        return 0, None



def load_yolo_annotations(annotation_path):
    """Load YOLO format annotation file"""
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    polygons = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        coords = list(map(float, parts[1:]))
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        polygons.append(points)
    
    return polygons



def get_adaptive_display_params(image_width, image_height):
    """Calculate adaptive display parameters - LARGER SIZES"""
    diagonal = np.sqrt(image_width**2 + image_height**2)
    base_diagonal = 2203
    scale_factor = diagonal / base_diagonal
    
    # Font scale - INCREASED for better visibility
    font_scale = 0.8 * scale_factor                 # Increased from 1.0 to 1.2
    font_scale = max(0.6, min(font_scale, 2.0))     # Increased max
    
    # Font thickness - INCREASED
    font_thickness = int(3 * scale_factor)
    font_thickness = max(2, min(font_thickness, 6))
    
    # Boundary thickness
    boundary_thickness = int(3 * scale_factor)
    boundary_thickness = max(2, min(boundary_thickness, 6))
    
    return font_scale, font_thickness, boundary_thickness



def create_normal_visualization(image_path, annotation_path, output_path):
    """
    Create NORMAL visualization with NO BRIGHTNESS LOSS
    Numbers drawn LAST on top of everything - NO yellow circles
    Returns cluster statistics for Excel
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, []
        
        h, w = img.shape[:2]
        font_scale, font_thickness, boundary_thickness = get_adaptive_display_params(w, h)
        
        polygons = load_yolo_annotations(annotation_path)
        if not polygons:
            return False, []
        
        # START WITH ORIGINAL IMAGE - 100% BRIGHTNESS PRESERVED
        result = img.copy()
        
        # Create separate masks for selective blending
        combined_flower_mask = np.zeros((h, w), dtype=np.uint8)
        combined_edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Customizable colors
        BOUNDARY_COLOR = (0, 255, 255)        # Yellow boundaries
        TEXT_OUTLINE_COLOR = (0, 0, 0)        # Black text outline
        TEXT_MAIN_COLOR = (255, 255, 255)     # White text
        OUTLINE_EXTRA_THICKNESS = 8           # THICK outline for maximum visibility
        
        cluster_stats = []
        cluster_centroids = []  # Store centroids to draw numbers LAST
        total_flower_pixels = 0
        total_edge_pixels = 0
        
        for i, polygon in enumerate(polygons):
            polygon_int = np.array([
                [int(x * w), int(y * h)] for x, y in polygon
            ], dtype=np.int32)
            
            # Calculate on ORIGINAL image (accurate detection)
            flower_pixels_count, flower_mask = calculate_enhanced_flower_pixel_count(img, polygon_int)
            edge_pixels_count, edge_mask = calculate_edge_pixels_in_flowers(img, flower_mask)
            
            # Accumulate masks
            if flower_mask is not None and np.any(flower_mask):
                combined_flower_mask = cv2.bitwise_or(combined_flower_mask, flower_mask)
                total_flower_pixels += flower_pixels_count
            
            if edge_mask is not None and np.any(edge_mask):
                combined_edge_mask = cv2.bitwise_or(combined_edge_mask, edge_mask)
                total_edge_pixels += edge_pixels_count
            
            cluster_stats.append({
                'cluster_id': i + 1,
                'flower_pixels': int(flower_pixels_count),
                'edge_pixels': int(edge_pixels_count)
            })
            
            # Draw boundaries on result
            cv2.polylines(result, [polygon_int], True, BOUNDARY_COLOR, boundary_thickness)
            
            # Store centroid for drawing numbers LATER (on top)
            centroid_x = int(np.mean(polygon_int[:, 0]))
            centroid_y = int(np.mean(polygon_int[:, 1]))
            cluster_centroids.append((i + 1, centroid_x, centroid_y))
        
        # ===== SELECTIVE BLENDING - PRESERVES ORIGINAL BRIGHTNESS =====
        alpha_flower = 0.5  # Adjust for flower visibility (0.3-0.7)
        alpha_edge = 0.3    # Adjust for edge visibility (0.2-0.5)
        
        if np.any(combined_flower_mask > 0):
            flower_locations = combined_flower_mask > 0
            result[flower_locations] = (
                img[flower_locations].astype(np.float32) * (1 - alpha_flower) + 
                np.array([0, 0, 255], dtype=np.float32) * alpha_flower
            ).astype(np.uint8)
        
        # Apply GREEN overlay ONLY on edge pixels
        if np.any(combined_edge_mask > 0):
            edge_locations = combined_edge_mask > 0
            result[edge_locations] = (
                img[edge_locations].astype(np.float32) * (1 - alpha_edge) + 
                np.array([0, 255, 0], dtype=np.float32) * alpha_edge
            ).astype(np.uint8)
        
        # ===== DRAW NUMBERS LAST - ON TOP OF EVERYTHING =====
        # NO yellow circles, just white numbers with THICK black outline
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for cluster_id, centroid_x, centroid_y in cluster_centroids:
            text = str(cluster_id)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            
            # THICK black outline (highly visible)
            cv2.putText(result, text, (text_x, text_y), font, font_scale, 
                       TEXT_OUTLINE_COLOR, font_thickness + OUTLINE_EXTRA_THICKNESS, cv2.LINE_AA)
            
            # White text on top
            cv2.putText(result, text, (text_x, text_y), font, font_scale, 
                       TEXT_MAIN_COLOR, font_thickness, cv2.LINE_AA)
        # ================================================================
        
        # Add title and stats
        title_font_scale = max(0.6, font_scale * 1.2)
        title_thickness = max(2, font_thickness)
        
        title = f"Flower Clusters: {len(polygons)}"
        title_size = cv2.getTextSize(title, font, title_font_scale, title_thickness)[0]
        cv2.rectangle(result, (5, 5), (title_size[0] + 20, 40), (0, 0, 0), -1)
        cv2.putText(result, title, (15, 30), font, title_font_scale, 
                   (255, 255, 255), title_thickness, cv2.LINE_AA)
        
        stats_font_scale = max(0.5, font_scale * 0.9)
        stats_text = f"Flowers: {total_flower_pixels} | Edges: {total_edge_pixels}"
        stats_size = cv2.getTextSize(stats_text, font, stats_font_scale, 1)[0]
        cv2.rectangle(result, (5, 50), (stats_size[0] + 20, 85), (0, 0, 0), -1)
        cv2.putText(result, stats_text, (15, 75), font, stats_font_scale, 
                   (255, 255, 255), 1, cv2.LINE_AA)
        
        # Legend
        legend_font_scale = max(0.4, font_scale * 0.7)
        legend_box_size = int(15 * (w / 1920))
        legend_box_size = max(10, min(legend_box_size, 20))
        legend_y = h - 80
        legend_width = int(350 * (w / 1920))
        legend_width = max(250, min(legend_width, 450))
        
        cv2.rectangle(result, (5, legend_y - 10), (legend_width, h - 5), (0, 0, 0), -1)
        
        cv2.rectangle(result, (15, legend_y), 
                     (15 + legend_box_size, legend_y + legend_box_size), (0, 0, 255), -1)
        cv2.putText(result, "= Flower Pixels", (15 + legend_box_size + 10, legend_y + legend_box_size - 3), 
                   font, legend_font_scale, (255, 255, 255), 1)
        
        cv2.rectangle(result, (15, legend_y + 25), 
                     (15 + legend_box_size, legend_y + 25 + legend_box_size), (0, 255, 0), -1)
        cv2.putText(result, "= Edge Pixels", (15 + legend_box_size + 10, legend_y + 25 + legend_box_size - 3), 
                   font, legend_font_scale, (255, 255, 255), 1)
        
        cv2.rectangle(result, (15, legend_y + 50), 
                     (15 + legend_box_size, legend_y + 50 + legend_box_size), BOUNDARY_COLOR, -1)
        cv2.putText(result, "= Boundaries", (15 + legend_box_size + 10, legend_y + 50 + legend_box_size - 3), 
                   font, legend_font_scale, (255, 255, 255), 1)
        
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True, cluster_stats
        
    except Exception as e:
        print(f"Error creating normal visualization: {e}")
        return False, []


def create_binary_edge_visualization(image_path, annotation_path, output_path):
    """
    Create BINARY visualization: WHITE = edge pixels, BLACK = everything else
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        h, w = img.shape[:2]
        polygons = load_yolo_annotations(annotation_path)
        
        if not polygons:
            return False
        
        # Pure BLACK background
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Combine all edge pixels
        all_edge_pixels = np.zeros((h, w), dtype=np.uint8)
        
        for polygon in polygons:
            polygon_int = np.array([
                [int(x * w), int(y * h)] for x, y in polygon
            ], dtype=np.int32)
            
            flower_pixels_count, flower_mask = calculate_enhanced_flower_pixel_count(img, polygon_int)
            
            if flower_mask is not None and np.any(flower_mask):
                edge_pixels_count, edge_mask = calculate_edge_pixels_in_flowers(img, flower_mask)
                
                if edge_mask is not None and np.any(edge_mask):
                    all_edge_pixels[edge_mask > 0] = 255
        
        # Set edge pixels to WHITE
        result[all_edge_pixels > 0] = [255, 255, 255]
        
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        print(f"Error creating binary visualization: {e}")
        return False



def export_single_image_excel(image_name, cluster_stats, dimensions, output_path):
    """Export Excel file for single image"""
    try:
        image_w, image_h = dimensions
        total_flower_pixels = sum([c['flower_pixels'] for c in cluster_stats])
        total_edge_pixels = sum([c['edge_pixels'] for c in cluster_stats])
        
        # Cluster details
        clusters_df = pd.DataFrame(cluster_stats)
        
        # Image summary
        summary_data = {
            'Metric': [
                'Image Name',
                'Image Width',
                'Image Height',
                'Total Clusters',
                'Total Flower Pixels',
                'Total Edge Pixels',
                'Avg Flower Pixels per Cluster',
                'Avg Edge Pixels per Cluster'
            ],
            'Value': [
                image_name,
                image_w,
                image_h,
                len(cluster_stats),
                total_flower_pixels,
                total_edge_pixels,
                round(total_flower_pixels / len(cluster_stats), 2) if cluster_stats else 0,
                round(total_edge_pixels / len(cluster_stats), 2) if cluster_stats else 0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            clusters_df.to_excel(writer, sheet_name='Cluster_Details', index=False)
            summary_df.to_excel(writer, sheet_name='Image_Summary', index=False)
        
        return True
        
    except Exception as e:
        print(f"Error exporting Excel: {e}")
        return False



def process_all_images_triple_output(images_folder, annotations_folder, base_output_folder):
    """
    Process all images and create 3 outputs per image in 3 separate folders
     NO BRIGHTNESS LOSS - Original image brightness preserved!
     Numbers drawn on TOP - always visible
     NO yellow circles - clean white numbers with black outline
    """
    print(" TRIPLE OUTPUT PROCESSOR - CLEAN NUMBERS VERSION")
    print("=" * 80)
    print("INPUT: Images + Annotations")
    print("OUTPUT PER IMAGE:")
    print("   Normal Visualization (flowers + edges + boundaries + numbers)")
    print("   Binary Edge Visualization (white edges on black)")
    print("   Individual Excel file")
    print("\n KEY FEATURES:")
    print("   100% ORIGINAL BRIGHTNESS MAINTAINED")
    print("   Numbers drawn LAST (always on top)")
    print("   NO yellow circles (clean white numbers)")
    print("   THICK black outline for maximum visibility")
    print("=" * 80)
    
    images_path = Path(images_folder)
    annotations_path = Path(annotations_folder)
    base_output_path = Path(base_output_folder)
    
    # Create 3 output folders
    normal_folder = base_output_path / "1_normal_visualization"
    binary_folder = base_output_path / "2_binary_edge_visualization"
    excel_folder = base_output_path / "3_excel_reports"
    
    normal_folder.mkdir(parents=True, exist_ok=True)
    binary_folder.mkdir(parents=True, exist_ok=True)
    excel_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Output folders:")
    print(f"   Normal: {normal_folder}")
    print(f"   Binary: {binary_folder}")
    print(f"   Excel: {excel_folder}\n")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f'*{ext}'))
        image_files.extend(images_path.glob(f'*{ext.upper()}'))
    
    print(f" Found {len(image_files)} images\n")
    
    processed = 0
    failed = 0
    
    for image_file in tqdm(image_files, desc=" Processing"):
        annotation_file = annotations_path / f"{image_file.stem}.txt"
        
        if not annotation_file.exists():
            print(f"  No annotation for {image_file.name}")
            failed += 1
            continue
        
        try:
            # Output paths
            normal_output = normal_folder / f"{image_file.stem}_normal.jpg"
            binary_output = binary_folder / f"{image_file.stem}_binary_edges.jpg"
            excel_output = excel_folder / f"{image_file.stem}_report.xlsx"
            
            # 1. Create normal visualization (NO BRIGHTNESS LOSS + CLEAN NUMBERS)
            success_normal, cluster_stats = create_normal_visualization(
                image_file, annotation_file, normal_output
            )
            
            if not success_normal:
                failed += 1
                continue
            
            # 2. Create binary edge visualization
            success_binary = create_binary_edge_visualization(
                image_file, annotation_file, binary_output
            )
            
            # 3. Export Excel
            img = cv2.imread(str(image_file))
            h, w = img.shape[:2]
            success_excel = export_single_image_excel(
                image_file.name, cluster_stats, (w, h), excel_output
            )
            
            if success_normal and success_binary and success_excel:
                processed += 1
            else:
                failed += 1
        
        except Exception as e:
            print(f" Error processing {image_file.name}: {e}")
            failed += 1
    
    print(f"\n PROCESSING COMPLETE!")
    print(f"=" * 80)
    print(f" Successfully processed: {processed}/{len(image_files)} images")
    print(f" Failed: {failed} images")
    print(f"\n Output Summary:")
    print(f"   Normal visualizations: {len(list(normal_folder.glob('*.jpg')))}")
    print(f"   Binary edge images: {len(list(binary_folder.glob('*.jpg')))}")
    print(f"   Excel reports: {len(list(excel_folder.glob('*.xlsx')))}")
    print(f"\n Final Features:")
    print(f"   NO BRIGHTNESS LOSS - 100% original preserved")
    print(f"   Clean white numbers (no yellow circles)")
    print(f"   Numbers always on top (never obscured)")
    print(f"   THICK black outline for visibility")
    print(f"   Selective overlay (only flower/edge pixels)")
    print(f"   Accurate calculations on original brightness")
    print(f"   Individual Excel per image")



# ==================== USAGE ====================


if __name__ == "__main__":
    
    print("=" * 80)
    print(" TRIPLE OUTPUT - CLEAN NUMBERS + NO BRIGHTNESS LOSS")
    print("=" * 80)
    print(" IMPROVEMENTS:")
    print(" Numbers drawn LAST (always visible on top)")
    print(" NO yellow circles (removed bright background)")
    print(" White numbers with THICK black outline")
    print(" 100% original brightness preserved")
    print("=" * 80)
    print()
    
    # Configuration
    IMAGES_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step4_clahe\output"
    ANNOTATIONS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step3_merge_annot\annotations"
    BASE_OUTPUT_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step5_flower_mask\output"
    
    process_all_images_triple_output(
        IMAGES_FOLDER,
        ANNOTATIONS_FOLDER,
        BASE_OUTPUT_FOLDER
    )
    
    print("\n ALL OUTPUTS GENERATED!")
    print(" Numbers are now clean, bright, and always on top!")
    print(" Check the 3 folders for your results!")
