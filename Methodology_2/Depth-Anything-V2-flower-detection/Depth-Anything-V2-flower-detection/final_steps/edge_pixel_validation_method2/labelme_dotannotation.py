import json
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, box
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ==================== LOADERS ====================

def load_yolo_bboxes(annotation_path: Path):
    """
    Load YOLO bbox annotations:
    class_id  x_center  y_center  width  height   (all normalized 0..1)
    Returns: list of tuples (class_id, xc, yc, w, h)
    """
    bboxes = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            bboxes.append((cls, xc, yc, w, h))
    return bboxes


def load_labelme_points(json_path: Path):
    """
    Load point annotations from LabelMe JSON.
    Returns: list of shapely Points (pixel coords)
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        points = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "point":
                pts = shape.get("points", [])
                if pts and len(pts) > 0:
                    x, y = pts[0]
                    points.append(Point(x, y))

        return points

    except Exception as e:
        print(f"Error loading LabelMe points from today's file: {json_path}\n  {e}")
        return []


def get_image_dimensions_from_labelme(json_path: Path):
    """
    Extract image dimensions from LabelMe JSON.
    Returns: (img_w, img_h)
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        img_w = int(data.get("imageWidth", 0))
        img_h = int(data.get("imageHeight", 0))
        return img_w, img_h
    except Exception:
        return 0, 0


def yolo_bboxes_to_shapely_rects(bbox_txt_path: Path, img_w: int, img_h: int):
    """
    Convert YOLO normalized bboxes to shapely rectangles (pixel coords).
    Returns: list of shapely geometries (rectangles) + list of class_ids
    """
    bboxes = load_yolo_bboxes(bbox_txt_path)
    rects = []
    class_ids = []

    for cls, xc, yc, w, h in bboxes:
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h

        rects.append(box(x1, y1, x2, y2))
        class_ids.append(cls)

    return rects, class_ids


# ==================== CORE ====================

def count_dots_per_box(bbox_txt_path: Path, points_json_path: Path, count_border_points: bool = True):
    """
    Count LabelMe dot points per YOLO bounding box.

    count_border_points:
      - True  => counts points on the rectangle edge too (uses rect.covers)
      - False => counts only strictly inside (uses rect.contains)

    Returns list of dicts:
      [{'box_id': 1, 'class_id': 0, 'dot_count': 5}, ...]
    """
    try:
        img_w, img_h = get_image_dimensions_from_labelme(points_json_path)
        if img_w == 0 or img_h == 0:
            print(f"   Could not get image dimensions from {points_json_path}")
            return []

        rects, class_ids = yolo_bboxes_to_shapely_rects(bbox_txt_path, img_w, img_h)
        points = load_labelme_points(points_json_path)

        print(f"   {len(rects)} boxes, {len(points)} dots")

        cluster_stats = []
        total_dots = 0

        for i, (rect, cls) in enumerate(zip(rects, class_ids), start=1):
            if count_border_points:
                dot_count = sum(1 for p in points if rect.covers(p))
            else:
                dot_count = sum(1 for p in points if rect.contains(p))

            total_dots += dot_count
            cluster_stats.append({
                "box_id": i,
                "class_id": cls,
                "dot_count": dot_count
            })

        print(f"   Total dots counted (sum across boxes): {total_dots}")
        return cluster_stats

    except Exception as e:
        print(f"   Error counting dots: {e}")
        import traceback
        traceback.print_exc()
        return []


def export_box_dots_excel(image_name: str, box_stats, output_path: Path):
    """
    Export Excel with:
      - Box_Dots sheet: box_id, class_id, dot_count
      - Summary sheet: totals and averages
    """
    try:
        boxes_df = pd.DataFrame(box_stats)

        total_boxes = len(box_stats)
        total_dots_sum_across_boxes = int(sum(b["dot_count"] for b in box_stats))
        avg_dots = round(total_dots_sum_across_boxes / total_boxes, 2) if total_boxes else 0

        boxes_with_dots = sum(1 for b in box_stats if b["dot_count"] > 0)
        boxes_without_dots = total_boxes - boxes_with_dots

        summary_df = pd.DataFrame({
            "Metric": [
                "Image Name",
                "Total Boxes",
                "Total Dots (sum across boxes)",
                "Average Dots per Box",
                "Boxes with Dots",
                "Boxes without Dots",
            ],
            "Value": [
                image_name,
                total_boxes,
                total_dots_sum_across_boxes,
                avg_dots,
                boxes_with_dots,
                boxes_without_dots,
            ],
        })

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            boxes_df.to_excel(writer, sheet_name="Box_Dots", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return True

    except Exception as e:
        print(f"   Error exporting Excel: {e}")
        return False


def process_all_images_dot_counting(bboxes_folder: str, labelme_folder: str, base_output_folder: str,
                                   count_border_points: bool = True):
    """
    Process all bbox .txt files:
      - find matching LabelMe .json (same stem)
      - count dots per bbox
      - export Excel per image
    """
    print(" DOT COUNTING PER YOLO BOUNDING BOX")
    print("=" * 80)
    print("INPUT:")
    print("  • Bounding box annotations (.txt YOLO bbox format)")
    print("  • Dot annotations (.json LabelMe points)")
    print("\nOUTPUT:")
    print("  • Excel files with box_id, class_id, dot_count")
    print("=" * 80)

    bboxes_path = Path(bboxes_folder)
    labelme_path = Path(labelme_folder)
    base_output_path = Path(base_output_folder)
    base_output_path.mkdir(parents=True, exist_ok=True)

    bbox_files = list(bboxes_path.glob("*.txt"))
    print(f"\n Output folder: {base_output_path}")
    print(f" Found {len(bbox_files)} bbox annotation files\n")

    processed = 0
    failed = 0

    for bbox_file in tqdm(bbox_files, desc=" Processing"):
        points_file = labelme_path / f"{bbox_file.stem}.json"

        if not points_file.exists():
            print(f"\n  No LabelMe point annotation for {bbox_file.name}")
            failed += 1
            continue

        try:
            print(f"\n Processing: {bbox_file.stem}")

            box_stats = count_dots_per_box(
                bbox_file,
                points_file,
                count_border_points=count_border_points
            )

            if not box_stats:
                failed += 1
                continue

            excel_output = base_output_path / f"{bbox_file.stem}_bbox_dots.xlsx"
            success = export_box_dots_excel(
                image_name=f"{bbox_file.stem}.jpg",
                box_stats=box_stats,
                output_path=excel_output
            )

            if success:
                processed += 1
                print(f"   Saved: {excel_output.name}")
            else:
                failed += 1

        except Exception as e:
            print(f"\n Error processing {bbox_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 80}")
    print(" PROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f" Successfully processed: {processed}/{len(bbox_files)} files")
    print(f" Failed: {failed} files")
    print(f" Total Excel files created: {len(list(base_output_path.glob('*.xlsx')))}")


# ==================== USAGE ====================

if __name__ == "__main__":
    print("=" * 80)
    print(" YOLO BBOX -> LABELME DOT COUNTER")
    print("=" * 80)
    print(" Outputs one Excel per image:")
    print("  • Box_Dots sheet: box_id, class_id, dot_count")
    print("  • Summary sheet: totals + averages")
    print("=" * 80)
    print()

    # CONFIGURE THESE PATHS
    BBOXES_FOLDER = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\edge_pixel_validation_method2\bb_annotations"
    LABELME_FOLDER = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\edge_pixel_validation_method2\labelme_annoataions"
    BASE_OUTPUT_FOLDER = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\edge_pixel_validation_method2\labelme_excel" 

    # If True, points exactly on bbox border are counted too.
    COUNT_BORDER_POINTS = True

    process_all_images_dot_counting(
        BBOXES_FOLDER,
        LABELME_FOLDER,
        BASE_OUTPUT_FOLDER,
        count_border_points=COUNT_BORDER_POINTS
    )

    print("\n DONE!")
    print(" Check output folder for Excel files with bbox dot counts.")
