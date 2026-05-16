import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN


class PolygonAnnotationMerger:
    """
    Standalone polygon merger using EXACT logic from YOLOSegmentationUncertaintyValidator.
    Input: Existing YOLO annotation files + images
    Output: Merged YOLO annotation files + visualization images
    """
    
    def __init__(self):
        """Initialize the merger"""
        pass
    
    # ========== EXACT MERGING PIPELINE FROM ORIGINAL CODE ==========
    
    def check_polygons_overlap_or_touch(self, poly1_coords, poly2_coords):
        try:
            poly1 = Polygon(poly1_coords)
            poly2 = Polygon(poly2_coords)
            
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            
            if poly1.intersects(poly2) or poly1.touches(poly2):
                return True
            
            for x1, y1 in poly1_coords:
                for x2, y2 in poly2_coords:
                    if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                        return True
            
            return False
        except Exception:
            return False
    
    def merge_touching_clusters(self, clusters, image_area, max_area_percentage=0.30):
        max_cluster_area = image_area * max_area_percentage
        merged_clusters = []
        merged_indices = set()
        
        def safe_create_polygon(coords):
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                return poly if poly.is_valid else None
            except Exception:
                return None
        
        def safe_union(poly1, poly2):
            try:
                if poly1 is None or poly2 is None:
                    return None
                result = poly1.union(poly2)
                if not result.is_valid:
                    result = result.buffer(0)
                return result if result.is_valid else None
            except Exception:
                return None
        
        for i, cluster in enumerate(clusters):
            if i in merged_indices:
                continue
            
            current_group = [cluster]
            current_indices = {i}
            
            for j, other_cluster in enumerate(clusters):
                if j in merged_indices or j == i:
                    continue
                
                touches_any = False
                for existing_cluster in current_group:
                    if self.check_polygons_overlap_or_touch(existing_cluster, other_cluster):
                        touches_any = True
                        break
                
                if touches_any:
                    existing_poly = safe_create_polygon(existing_cluster)
                    other_poly = safe_create_polygon(other_cluster)
                    
                    if existing_poly and other_poly:
                        combined_poly = safe_union(existing_poly, other_poly)
                        if combined_poly and combined_poly.area <= max_cluster_area:
                            current_group.append(other_cluster)
                            current_indices.add(j)
            
            merged_indices.update(current_indices)
            
            if len(current_group) > 1:
                try:
                    valid_polys = []
                    for coords in current_group:
                        poly = safe_create_polygon(coords)
                        if poly:
                            valid_polys.append(poly)
                    
                    if not valid_polys:
                        merged_clusters.extend(current_group)
                        continue
                    
                    combined = unary_union(valid_polys)
                    
                    if not combined.is_valid:
                        combined = combined.buffer(0)
                    
                    if combined.area <= max_cluster_area:
                        if combined.geom_type == 'Polygon':
                            merged_clusters.append(list(combined.exterior.coords[:-1]))
                        elif combined.geom_type == 'MultiPolygon':
                            for geom in combined.geoms:
                                merged_clusters.append(list(geom.exterior.coords[:-1]))
                        else:
                            merged_clusters.extend(current_group)
                    else:
                        merged_clusters.extend(current_group)
                except Exception:
                    merged_clusters.extend(current_group)
            else:
                merged_clusters.append(cluster)
        
        return merged_clusters
    
    def aggressive_cluster_flower_annotations(self, polygons, image_shape, target_clusters=15):
        if len(polygons) <= target_clusters:
            return [[poly] for poly in polygons]
        
        img_h, img_w = image_shape[:2]
        image_diagonal = np.sqrt(img_w**2 + img_h**2)
        
        centroids = np.array([[np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])] for poly in polygons])
        
        areas = []
        for poly in polygons:
            try:
                if len(poly) >= 3:
                    p = Polygon(poly)
                    if not p.is_valid:
                        p = p.buffer(0)
                    areas.append(p.area if p.is_valid else 0)
                else:
                    areas.append(0)
            except:
                areas.append(0)
        
        epsilon_values = [0.15, 0.20, 0.25, 0.30]
        
        for eps_factor in epsilon_values:
            epsilon = image_diagonal * eps_factor
            clustering = DBSCAN(eps=epsilon, min_samples=1, metric='euclidean')
            labels = clustering.fit_predict(centroids)
            
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if target_clusters - 3 <= num_clusters <= target_clusters + 3:
                clusters = [[] for _ in range(max(labels) + 1)]
                for idx, label in enumerate(labels):
                    if label >= 0:
                        clusters[label].append(polygons[idx])
                return [cluster for cluster in clusters if cluster]
        
        clusters = [[] for _ in range(max(labels) + 1)]
        for idx, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(polygons[idx])
        
        return [cluster for cluster in clusters if cluster]
    
    def combine_cluster_polygons(self, cluster_polygons, image_area, max_area_percentage=0.15):
        max_cluster_area = image_area * max_area_percentage
        
        try:
            valid_polygons = []
            for poly_coords in cluster_polygons:
                if len(poly_coords) >= 3:
                    try:
                        poly = Polygon(poly_coords)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_valid:
                            valid_polygons.append(poly)
                    except Exception:
                        continue
            
            if not valid_polygons:
                return cluster_polygons
            
            combined = unary_union(valid_polygons)
            
            if not combined.is_valid:
                combined = combined.buffer(0)
            
            if combined.area > max_cluster_area:
                return cluster_polygons
            
            if combined.geom_type == 'Polygon':
                return [list(combined.exterior.coords[:-1])]
            elif combined.geom_type == 'MultiPolygon':
                return [list(geom.exterior.coords[:-1]) for geom in combined.geoms]
            else:
                return cluster_polygons
                
        except Exception:
            return cluster_polygons
    
    def apply_full_merging_pipeline(self, raw_polygons, image_shape):
        if not raw_polygons:
            return []
        
        h, w = image_shape[:2]
        image_area = h * w
        
        clustered = self.aggressive_cluster_flower_annotations(raw_polygons, image_shape, target_clusters=8)
        
        combined_clusters = []
        for cluster_polys in clustered:
            combined = self.combine_cluster_polygons(cluster_polys, image_area, max_area_percentage=0.15)
            combined_clusters.extend(combined)
        
        merged_once = self.merge_touching_clusters(combined_clusters, image_area, max_area_percentage=0.30)
        final_merged = self.merge_touching_clusters(merged_once, image_area, max_area_percentage=0.30)
        
        return final_merged
    
    # ========== END EXACT MERGING PIPELINE ==========
    
    # ========== YOLO FORMAT CONVERSION ==========
    
    def yolo_to_polygon(self, yolo_line, img_width, img_height):
        """Convert YOLO format line to polygon pixel coordinates"""
        try:
            parts = yolo_line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                return None
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            if len(coords) % 2 != 0:
                return None
            
            # Convert normalized to pixel coordinates
            polygon = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_width
                y = coords[i + 1] * img_height
                polygon.append([x, y])
            
            return class_id, polygon
        except Exception:
            return None
    
    def polygon_to_yolo_format(self, polygon, img_width, img_height):
        """Convert polygon pixel coordinates to YOLO format (exact same as original)"""
        if polygon is None or len(polygon) == 0:
            return None
        
        normalized_coords = []
        for point in polygon:
            x = max(0.0, min(1.0, float(point[0]) / img_width))
            y = max(0.0, min(1.0, float(point[1]) / img_height))
            normalized_coords.extend([x, y])
        
        if len(normalized_coords) < 6:
            return None
            
        return normalized_coords
    
    # ========== FILE HANDLING ==========
    
    def get_unique_images(self, folder_path):
        """Get unique image files (exact same as original)"""
        folder = Path(folder_path)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_files = []
        
        for ext in extensions:
            all_files.extend(folder.glob(f'*{ext}'))
            all_files.extend(folder.glob(f'*{ext.upper()}'))
        
        unique_basenames = set()
        unique_files = []
        for f in all_files:
            base_name = f.stem.lower()
            if base_name not in unique_basenames:
                unique_basenames.add(base_name)
                unique_files.append(f)
        
        return unique_files
    
    # ========== VISUALIZATION ==========
    
    def draw_polygons_on_image(self, image_path, merged_polygons, output_path):
        """Draw merged polygons on image and save"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Draw each merged polygon
            for polygon in merged_polygons:
                if len(polygon) >= 3:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(img, [pts], isClosed=True, 
                                color=(0, 255, 0), thickness=2)
            
            cv2.imwrite(str(output_path), img)
            return True
        except Exception:
            return False
    
    # ========== MAIN PROCESSING ==========
    
    def process_annotations(self, images_folder, input_annotations_folder, 
                          output_annotations_folder, output_visualizations_folder=None):
        """
        Process folder: read existing annotations, apply merging, save merged annotations + visualizations
        
        Args:
            images_folder: Folder containing images
            input_annotations_folder: Folder containing original YOLO annotations (.txt files)
            output_annotations_folder: Folder to save merged annotations
            output_visualizations_folder: Optional folder to save visualization images with polygons drawn
        """
        os.makedirs(output_annotations_folder, exist_ok=True)
        
        if output_visualizations_folder:
            os.makedirs(output_visualizations_folder, exist_ok=True)
        
        image_files = self.get_unique_images(images_folder)
        
        print(f" Found {len(image_files)} images")
        print(f" Processing annotations with EXACT merging pipeline...")
        
        processed_count = 0
        skipped_count = 0
        
        for image_path in tqdm(image_files, desc="Merging annotations"):
            try:
                # Load image to get dimensions
                img_pil = Image.open(image_path)
                img_width, img_height = img_pil.size
                
                # Find corresponding annotation file
                annotation_path = Path(input_annotations_folder) / f"{image_path.stem}.txt"
                
                if not annotation_path.exists():
                    # Create empty annotation file
                    output_ann_path = Path(output_annotations_folder) / f"{image_path.stem}.txt"
                    with open(output_ann_path, 'w') as f:
                        pass
                    skipped_count += 1
                    continue
                
                # Read and parse annotations
                raw_polygons = []
                
                with open(annotation_path, 'r') as f:
                    for line in f:
                        result = self.yolo_to_polygon(line, img_width, img_height)
                        if result:
                            class_id, polygon = result
                            if len(polygon) >= 3:
                                raw_polygons.append(polygon)
                
                if not raw_polygons:
                    # Create empty annotation file
                    output_ann_path = Path(output_annotations_folder) / f"{image_path.stem}.txt"
                    with open(output_ann_path, 'w') as f:
                        pass
                    skipped_count += 1
                    continue
                
                # Apply EXACT merging pipeline
                merged_polygons = self.apply_full_merging_pipeline(
                    raw_polygons, (img_height, img_width)
                )
                
                # Save merged annotations in YOLO format
                output_ann_path = Path(output_annotations_folder) / f"{image_path.stem}.txt"
                detections = []
                
                for merged_poly in merged_polygons:
                    polygon_coords = self.polygon_to_yolo_format(
                        merged_poly, img_width, img_height
                    )
                    
                    if polygon_coords and len(polygon_coords) >= 6:
                        coords_str = ' '.join([f"{coord:.6f}" for coord in polygon_coords])
                        detections.append(f"0 {coords_str}")
                
                with open(output_ann_path, 'w') as f:
                    f.write('\n'.join(detections))
                
                # Generate visualization if requested
                if output_visualizations_folder:
                    vis_path = Path(output_visualizations_folder) / f"{image_path.stem}.jpg"
                    self.draw_polygons_on_image(image_path, merged_polygons, vis_path)
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n  Error processing {image_path.name}: {str(e)}")
                skipped_count += 1
                continue
        
        print(f"\n Processing complete!")
        print(f"   Processed: {processed_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Output annotations: {output_annotations_folder}")
        if output_visualizations_folder:
            print(f"   Output visualizations: {output_visualizations_folder}")


def main():
    """Example usage"""
    
    # ========== CONFIGURATION ==========
    IMAGES_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step 1.3_imgproc\output"
    INPUT_ANNOTATIONS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step2_yolo\output\annotations"  # Your existing annotations
    OUTPUT_ANNOTATIONS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step3_merge_annot\annotations"
    OUTPUT_VISUALIZATIONS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step3_merge_annot\output"
    
    print("="*60)
    print("POLYGON ANNOTATION MERGER (EXACT LOGIC)")
    print("="*60)
    
    # Initialize merger
    merger = PolygonAnnotationMerger()
    
    # Process annotations
    merger.process_annotations(
        images_folder=IMAGES_FOLDER,
        input_annotations_folder=INPUT_ANNOTATIONS_FOLDER,
        output_annotations_folder=OUTPUT_ANNOTATIONS_FOLDER,
        output_visualizations_folder=OUTPUT_VISUALIZATIONS_FOLDER  # Set to None if you don't want visualizations
    )
    
    print("\n COMPLETE!")


if __name__ == "__main__":
    main()
