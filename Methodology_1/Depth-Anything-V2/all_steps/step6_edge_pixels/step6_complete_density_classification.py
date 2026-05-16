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




def classify_density(edge_pixels, all_edge_values=None):
    """
    Classify cluster density based on edge pixel count using fixed thresholds.
    Returns: 'Very Low', 'Low', 'Medium', or 'High'
    
    Thresholds:
        < 50     -> Very Low
        50–499   -> Low
        500–1099 -> Medium
        1100+    -> High
    """
    if edge_pixels < 50:
        return 'Very Low'
    elif 50 <= edge_pixels < 500:
        return 'Low'
    elif 500 <= edge_pixels < 1100:
        return 'Medium'
    else:
        return 'High'





def get_density_color(density):
    """
    Return BGR color based on density classification.
    Blue = Very Low, Green = Low, Yellow = Medium, Red = High
    """
    if density == 'Very Low':
        return (255, 0, 0)      # Blue
    elif density == 'Low':
        return (0, 255, 0)      # Green
    elif density == 'Medium':
        return (0, 255, 255)    # Yellow
    else:  # High
        return (0, 0, 255)      # Red





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
    """Calculate adaptive display parameters"""
    diagonal = np.sqrt(image_width**2 + image_height**2)
    base_diagonal = 2203
    scale_factor = diagonal / base_diagonal
    
    # Font scale
    font_scale = 0.8 * scale_factor
    font_scale = max(0.6, min(font_scale, 2.0))
    
    # Font thickness
    font_thickness = int(3 * scale_factor)
    font_thickness = max(2, min(font_thickness, 6))
    
    # Boundary thickness
    boundary_thickness = int(3 * scale_factor)
    boundary_thickness = max(2, min(boundary_thickness, 6))
    
    return font_scale, font_thickness, boundary_thickness




def create_density_visualization(image_path, annotation_path, output_path):
    """
    Create visualization with color-coded boundaries based on density
    Green = Low, Yellow = Medium, Red = High
    NO TOP INFO, NO BOTTOM LEGEND - CLEAN IMAGE
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
        
        # Text styling
        TEXT_OUTLINE_COLOR = (0, 0, 0)        # Black text outline
        TEXT_MAIN_COLOR = (255, 255, 255)     # White text
        OUTLINE_EXTRA_THICKNESS = 8
        
        # First pass: Calculate all edge pixels to determine thresholds
        all_edge_values = []
        temp_data = []
        
        for i, polygon in enumerate(polygons):
            polygon_int = np.array([
                [int(x * w), int(y * h)] for x, y in polygon
            ], dtype=np.int32)
            
            flower_pixels_count, flower_mask = calculate_enhanced_flower_pixel_count(img, polygon_int)
            edge_pixels_count, edge_mask = calculate_edge_pixels_in_flowers(img, flower_mask)
            
            all_edge_values.append(edge_pixels_count)
            
            temp_data.append({
                'polygon': polygon_int,
                'flower_pixels': flower_pixels_count,
                'edge_pixels': edge_pixels_count,
                'flower_mask': flower_mask,
                'edge_mask': edge_mask
            })
        
        # Second pass: Classify and draw with appropriate colors
        cluster_stats = []
        cluster_centroids = []
        total_flower_pixels = 0
        total_edge_pixels = 0
        
        for i, data in enumerate(temp_data):
            polygon_int = data['polygon']
            flower_pixels_count = data['flower_pixels']
            edge_pixels_count = data['edge_pixels']
            flower_mask = data['flower_mask']
            edge_mask = data['edge_mask']
            
            # Classify density
            density = classify_density(edge_pixels_count, all_edge_values)
            
            # Get color for this density
            boundary_color = get_density_color(density)
            
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
                'edge_pixels': int(edge_pixels_count),
                'classification': density
            })
            
            # Draw boundaries with density color
            cv2.polylines(result, [polygon_int], True, boundary_color, boundary_thickness)
            
            # Store centroid for drawing numbers LATER
            centroid_x = int(np.mean(polygon_int[:, 0]))
            centroid_y = int(np.mean(polygon_int[:, 1]))
            cluster_centroids.append((i + 1, centroid_x, centroid_y))
        
        # ===== SELECTIVE BLENDING - PRESERVES ORIGINAL BRIGHTNESS =====
        alpha_flower = 0.5
        alpha_edge = 0.3
        
        if np.any(combined_flower_mask > 0):
            flower_locations = combined_flower_mask > 0
            result[flower_locations] = (
                img[flower_locations].astype(np.float32) * (1 - alpha_flower) + 
                np.array([0, 0, 255], dtype=np.float32) * alpha_flower
            ).astype(np.uint8)
        
        if np.any(combined_edge_mask > 0):
            edge_locations = combined_edge_mask > 0
            result[edge_locations] = (
                img[edge_locations].astype(np.float32) * (1 - alpha_edge) + 
                np.array([0, 255, 0], dtype=np.float32) * alpha_edge
            ).astype(np.uint8)
        
        # ===== DRAW NUMBERS LAST - ON TOP OF EVERYTHING =====
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for cluster_id, centroid_x, centroid_y in cluster_centroids:
            text = str(cluster_id)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            
            # THICK black outline
            cv2.putText(result, text, (text_x, text_y), font, font_scale, 
                       TEXT_OUTLINE_COLOR, font_thickness + OUTLINE_EXTRA_THICKNESS, cv2.LINE_AA)
            
            # White text on top
            cv2.putText(result, text, (text_x, text_y), font, font_scale, 
                       TEXT_MAIN_COLOR, font_thickness, cv2.LINE_AA)
        
        # ===== NO TOP INFO, NO BOTTOM LEGEND - COMPLETELY CLEAN =====
        
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True, cluster_stats
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False, []




def export_single_image_excel(image_name, cluster_stats, dimensions, output_path):
    """Export Excel file with density classification"""
    try:
        image_w, image_h = dimensions
        total_flower_pixels = sum([c['flower_pixels'] for c in cluster_stats])
        total_edge_pixels = sum([c['edge_pixels'] for c in cluster_stats])
        
        # Count densities
        low_count = sum(1 for c in cluster_stats if c['classification'] == 'Low')
        medium_count = sum(1 for c in cluster_stats if c['classification'] == 'Medium')
        high_count = sum(1 for c in cluster_stats if c['classification'] == 'High')
        
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
                'Avg Edge Pixels per Cluster',
                '',
                'Low Density Clusters',
                'Medium Density Clusters',
                'High Density Clusters'
            ],
            'Value': [
                image_name,
                image_w,
                image_h,
                len(cluster_stats),
                total_flower_pixels,
                total_edge_pixels,
                round(total_flower_pixels / len(cluster_stats), 2) if cluster_stats else 0,
                round(total_edge_pixels / len(cluster_stats), 2) if cluster_stats else 0,
                '',
                low_count,
                medium_count,
                high_count
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




def process_all_images_density_classification(images_folder, annotations_folder, base_output_folder):
    """
    Process all images with density classification
    Two outputs: Clean Visualization (NO text boxes) + Excel
    """
    print(" DENSITY CLASSIFICATION PROCESSOR - CLEAN VERSION")
    print("=" * 80)
    print("INPUT: Images + Annotations")
    print("OUTPUT PER IMAGE:")
    print("    Clean Density Visualization (NO top info, NO bottom legend)")
    print("    Excel Report (cluster details + classification)")
    print("\n DENSITY CLASSIFICATION (fixed thresholds):")
    print("   BLUE boundary   = Very Low (< 50 edges)")
    print("   GREEN boundary  = Low (50 - 499 edges)")
    print("   YELLOW boundary = Medium (500 - 1099 edges)")
    print("   RED boundary    = High ( >= 1100 edges)")

    print("=" * 80)
    
    images_path = Path(images_folder)
    annotations_path = Path(annotations_folder)
    base_output_path = Path(base_output_folder)
    
    # Create 2 output folders
    viz_folder = base_output_path / "density_visualizations"
    excel_folder = base_output_path / "excel_reports"
    
    viz_folder.mkdir(parents=True, exist_ok=True)
    excel_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Output folders:")
    print(f"    Visualizations: {viz_folder}")
    print(f"    Excel Reports: {excel_folder}\n")
    
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
            viz_output = viz_folder / f"{image_file.stem}_density.jpg"
            excel_output = excel_folder / f"{image_file.stem}_report.xlsx"
            
            # 1. Create density visualization (CLEAN - NO TEXT BOXES)
            success_viz, cluster_stats = create_density_visualization(
                image_file, annotation_file, viz_output
            )
            
            if not success_viz:
                failed += 1
                continue
            
            # 2. Export Excel
            img = cv2.imread(str(image_file))
            h, w = img.shape[:2]
            success_excel = export_single_image_excel(
                image_file.name, cluster_stats, (w, h), excel_output
            )
            
            if success_viz and success_excel:
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
    print(f"    Clean density visualizations: {len(list(viz_folder.glob('*.jpg')))}")
    print(f"    Excel reports: {len(list(excel_folder.glob('*.xlsx')))}")
    print(f"\n Features:")
    print(f"   Automatic density classification (Low/Medium/High)")
    print(f"   Color-coded boundaries (Green/Yellow/Red)")
    print(f"   COMPLETELY CLEAN - NO text overlays")
    print(f"   100% original brightness preserved")
    print(f"   Clean white numbers with black outline")
    print(f"   Individual Excel per image with classification")




# ==================== USAGE ====================



if __name__ == "__main__":
    
    print("=" * 80)
    print(" DENSITY CLASSIFICATION - CLEAN VERSION (NO OVERLAYS)")
    print("=" * 80)
    print(" FEATURES:")
    print(" Edge-based density classification (Low/Medium/High)")
    print(" Color-coded boundaries (Green/Yellow/Red)")
    print(" CLEAN output - NO top info, NO bottom legend")
    print("Excel with cluster details + classification")
    print(" 100% original brightness preserved")
    print(" Clean numbers always on top")
    print("=" * 80)
    print()
    
    # Configuration
    IMAGES_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step4_clahe\output"
    ANNOTATIONS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step3_merge_annot\annotations"
    BASE_OUTPUT_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step6_edge_pixels\output"
    
    process_all_images_density_classification(
        IMAGES_FOLDER,
        ANNOTATIONS_FOLDER,
        BASE_OUTPUT_FOLDER
    )
    
    print("\n ALL OUTPUTS GENERATED!")
    print(" Clean visualizations with color-coded density boundaries!")
    print(" Check the 2 folders for your results!")
