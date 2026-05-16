import json
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def load_yolo_polygons_normalized(txt_path):
    """
    Load YOLO polygon annotations (normalized coords).
    Format per line:
        <class_id> x1 y1 x2 y2 x3 y3 ...
    """
    polygons = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(pts)

    return polygons


def load_labelme_points(json_path):
    """Load LabelMe point annotations (pixel coords)."""
    points = []
    with open(json_path, "r") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "point":
            x, y = shape["points"][0]
            points.append(Point(x, y))

    return points


def get_image_size(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("imageWidth", 0), data.get("imageHeight", 0)


def convert_polygons_to_pixel(polygons_norm, w, h):
    return [Polygon([(x * w, y * h) for x, y in poly]) for poly in polygons_norm]


def count_points(points, polygons, include_boundary=False):
    inside = 0
    for p in points:
        if include_boundary:
            if any(poly.covers(p) for poly in polygons):
                inside += 1
        else:
            if any(poly.contains(p) for poly in polygons):
                inside += 1

    total = len(points)
    outside = total - inside
    return total, inside, outside


def process_all_images_to_excel(clusters_dir, labelme_dir, output_excel, include_boundary=False):
    clusters_dir = Path(clusters_dir)
    labelme_dir = Path(labelme_dir)
    output_excel = Path(output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    cluster_files = sorted(clusters_dir.glob("*.txt"))

    rows = []

    for txt_file in tqdm(cluster_files, desc="Processing"):
        stem = txt_file.stem
        json_file = labelme_dir / f"{stem}.json"

        if not json_file.exists():
            continue

        w, h = get_image_size(json_file)
        if w == 0 or h == 0:
            continue

        polygons_norm = load_yolo_polygons_normalized(txt_file)
        polygons = convert_polygons_to_pixel(polygons_norm, w, h)
        points = load_labelme_points(json_file)

        total, inside, outside = count_points(points, polygons, include_boundary)

        rows.append({
            "image_name": stem,
            "total_dots": total,
            "dots_inside_polygons": inside,
            "dots_outside_polygons": outside
        })

    df = pd.DataFrame(rows)

    # Optional summary row at the bottom
    summary = {
        "image_name": "TOTAL",
        "total_dots": df["total_dots"].sum(),
        "dots_inside_polygons": df["dots_inside_polygons"].sum(),
        "dots_outside_polygons": df["dots_outside_polygons"].sum()
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    df.to_excel(output_excel, index=False)
    print(f"\nExcel file saved: {output_excel}")


if __name__ == "__main__":

    CLUSTERS_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\flower_noflower_validation\yolo_annotations"
    LABELME_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\flower_noflower_validation\labelme_annoataions"
    OUTPUT_EXCEL = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\flower_noflower_validation\all_images_dot_summary.xlsx"

    process_all_images_to_excel(
        CLUSTERS_FOLDER,
        LABELME_FOLDER,
        OUTPUT_EXCEL,
        include_boundary=False  # set True to count boundary dots as inside
    )

    print("DONE")
