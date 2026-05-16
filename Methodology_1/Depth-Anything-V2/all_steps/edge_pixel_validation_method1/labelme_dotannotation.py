import json
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_yolo_annotations(annotation_path):
    """Load YOLO format annotation file - returns normalized polygons"""
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


def load_labelme_flowers(json_path):
    """Load flower points from LabelMe JSON - returns shapely Points (pixel coords)"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        flower_points = []
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'point':
                pts = shape.get('points', [])
                if pts and len(pts) > 0:
                    x, y = pts[0]
                    flower_points.append(Point(x, y))
        
        return flower_points
        
    except Exception as e:
        print(f"Error loading LabelMe flowers: {e}")
        return []


def load_cluster_polygons_shapely(annotation_path, img_w, img_h):
    """Convert YOLO normalized polygons to shapely Polygons in pixel coords"""
    polygons_norm = load_yolo_annotations(annotation_path)
    cluster_polys = []
    
    for poly in polygons_norm:
        pts_px = [(x * img_w, y * img_h) for x, y in poly]
        cluster_polys.append(Polygon(pts_px))
    
    return cluster_polys


def get_image_dimensions_from_labelme(json_path):
    """Extract image dimensions from LabelMe JSON"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_w = data.get('imageWidth', 0)
        img_h = data.get('imageHeight', 0)
        
        return img_w, img_h
    except:
        return 0, 0


def count_dots_per_cluster(cluster_txt_path, flower_json_path):
    """
    Count flower dots in each cluster
    Returns list of dicts: [{'cluster_id': 1, 'dot_count': 5}, ...]
    """
    try:
        # Get image dimensions from LabelMe JSON
        img_w, img_h = get_image_dimensions_from_labelme(flower_json_path)
        
        if img_w == 0 or img_h == 0:
            print(f"   Could not get image dimensions from {flower_json_path}")
            return []
        
        # Load cluster polygons (YOLO -> shapely)
        cluster_polys = load_cluster_polygons_shapely(cluster_txt_path, img_w, img_h)
        
        # Load flower points (LabelMe -> shapely)
        flower_points = load_labelme_flowers(flower_json_path)
        
        print(f"   {len(cluster_polys)} clusters, {len(flower_points)} flower dots")
        
        # Count dots per cluster
        cluster_stats = []
        total_dots = 0
        
        for i, cluster_poly in enumerate(cluster_polys, start=1):
            dot_count = sum(1 for p in flower_points if cluster_poly.contains(p))
            total_dots += dot_count
            
            cluster_stats.append({
                'cluster_id': i,
                'dot_count': dot_count
            })
        
        print(f"   Total dots counted: {total_dots}")
        
        return cluster_stats
        
    except Exception as e:
        print(f"   Error counting dots: {e}")
        import traceback
        traceback.print_exc()
        return []


def export_cluster_dots_excel(image_name, cluster_stats, output_path):
    """Export simple Excel with cluster_id and dot_count"""
    try:
        # Cluster details
        clusters_df = pd.DataFrame(cluster_stats)
        
        # Summary stats
        total_clusters = len(cluster_stats)
        total_dots = sum([c['dot_count'] for c in cluster_stats])
        avg_dots = round(total_dots / total_clusters, 2) if total_clusters > 0 else 0
        
        clusters_with_dots = sum(1 for c in cluster_stats if c['dot_count'] > 0)
        clusters_without_dots = total_clusters - clusters_with_dots
        
        summary_data = {
            'Metric': [
                'Image Name',
                'Total Clusters',
                'Total Flower Dots',
                'Average Dots per Cluster',
                'Clusters with Dots',
                'Clusters without Dots'
            ],
            'Value': [
                image_name,
                total_clusters,
                total_dots,
                avg_dots,
                clusters_with_dots,
                clusters_without_dots
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            clusters_df.to_excel(writer, sheet_name='Cluster_Dots', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return True
        
    except Exception as e:
        print(f"   Error exporting Excel: {e}")
        return False


def process_all_images_dot_counting(clusters_folder, labelme_folder, base_output_folder):
    """
    Process all images: count dots per cluster and export to Excel
    """
    print(" FLOWER DOT COUNTING PER CLUSTER")
    print("=" * 80)
    print("INPUT:")
    print("  • Cluster annotations (.txt YOLO format)")
    print("  • Flower dots (.json LabelMe format)")
    print("\nOUTPUT:")
    print("  • Excel files with cluster_id and dot_count")
    print("=" * 80)
    
    clusters_path = Path(clusters_folder)
    labelme_path = Path(labelme_folder)
    base_output_path = Path(base_output_folder)
    
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Output folder: {base_output_path}\n")
    
    # Find all cluster annotation files
    cluster_files = list(clusters_path.glob('*.txt'))
    
    print(f" Found {len(cluster_files)} cluster annotation files\n")
    
    processed = 0
    failed = 0
    
    for cluster_file in tqdm(cluster_files, desc=" Processing"):
        flower_file = labelme_path / f"{cluster_file.stem}.json"
        
        if not flower_file.exists():
            print(f"\n  No flower annotation for {cluster_file.name}")
            failed += 1
            continue
        
        try:
            print(f"\n Processing: {cluster_file.stem}")
            
            # Count dots per cluster
            cluster_stats = count_dots_per_cluster(cluster_file, flower_file)
            
            if not cluster_stats:
                failed += 1
                continue
            
            # Export to Excel
            excel_output = base_output_path / f"{cluster_file.stem}_dots.xlsx"
            success = export_cluster_dots_excel(
                f"{cluster_file.stem}.jpg",
                cluster_stats,
                excel_output
            )
            
            if success:
                processed += 1
                print(f"   Saved: {excel_output.name}")
            else:
                failed += 1
        
        except Exception as e:
            print(f"\n Error processing {cluster_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f" PROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f" Successfully processed: {processed}/{len(cluster_files)} files")
    print(f" Failed: {failed} files")
    print(f"\n Total Excel files created: {len(list(base_output_path.glob('*.xlsx')))}")


# ==================== USAGE ====================

if __name__ == "__main__":
    
    print("=" * 80)
    print(" SIMPLE FLOWER DOT COUNTER")
    print("=" * 80)
    print(" OUTPUTS:")
    print("• Excel with cluster_id and dot_count")
    print("Summary sheet with totals and averages")
    print("=" * 80)
    print()
    
    #  CONFIGURE THESE PATHS
    CLUSTERS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\edge_pixel_validation\yolo_annotations"      # YOLO .txt cluster annotations
    LABELME_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\edge_pixel_validation\labelme_annoataions"               # LabelMe .json flower dots
    BASE_OUTPUT_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\edge_pixel_validation\labelme_excel"    # Output Excel files
    
    process_all_images_dot_counting(
        CLUSTERS_FOLDER,
        LABELME_FOLDER,
        BASE_OUTPUT_FOLDER
    )
    
    print("\n DONE!")
    print(" Check output folder for Excel files with cluster dot counts!")
