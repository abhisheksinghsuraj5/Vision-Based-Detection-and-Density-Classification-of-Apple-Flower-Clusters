# step4_anomaly_cluster.py
import cv2
import numpy as np
from pathlib import Path
import csv
from collections import Counter

# === Folders ===
INPUT_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step3\output"
ORIGINAL_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1_depth_anything\Image"
OUTPUT_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step4\output"

# === Parameters ===
ASPECT_RATIO_THRESHOLD = 2.0
EDGE_DENSITY_THRESHOLD = 0.1999
CANNY_LOWER = 200
CANNY_UPPER = 255
MAX_DISTANCE = 140
CIRCULARITY_THRESHOLD = 0.8

# Cluster size thresholds (total edge pixels)
SMALL_EDGE_THRESHOLD = 500
MEDIUM_EDGE_THRESHOLD = 1500

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ORIGINAL_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# === FIXED 3 COLORS FOR SIZE CATEGORIES ===
# BGR format: (B, G, R)
SIZE_COLORS = {
    "Small":  (0, 255, 255),   # Yellow
    "Medium": (0, 255, 0),     # Green
    "Large":  (0, 140, 255)    # Orange
}

# --- Setup ---
in_dir = Path(INPUT_DIR)
orig_dir = Path(ORIGINAL_DIR)
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
annot_dir = out_dir / "annotations"
annot_dir.mkdir(parents=True, exist_ok=True)

# CSV: per-cluster details
csv_path = out_dir / "cluster_detailed_summary.csv"
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "image_name", "cluster_id", "size_category", "total_edge_pixels",
    "num_contours", "num_circular", "centroid_x", "centroid_y"
])

# Global summary
summary_stats = []

# Process each mask
files = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in VALID_EXTS])

if not files:
    print(f"No mask files found in: {in_dir}")
else:
    for filled_mask_path in files:
        filled_mask = cv2.imread(str(filled_mask_path), cv2.IMREAD_GRAYSCALE)
        if filled_mask is None:
            print(f"[skip] Cannot read mask: {filled_mask_path}")
            continue

        stem = filled_mask_path.stem
        original_path = None
        for ext in ORIGINAL_EXT:
            candidate = orig_dir / f"{stem}{ext}"
            if candidate.exists():
                original_path = candidate
                break

        if original_path is None:
            print(f"[skip] Original image not found for stem '{stem}' in {orig_dir}")
            continue
            
        original_img = cv2.imread(str(original_path))
        if original_img is None:
            print(f"[skip] Original image not found: {original_path}")
            continue

        # --- Find Contours ---
        _, bin_mask = cv2.threshold(filled_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Anomaly Filtering ---
        filtered_contours = []
        filtered_edge_pixels = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w != 0 else float('inf')

            mask = np.zeros_like(filled_mask)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            total_mask_pixels = cv2.contourArea(cnt)

            roi = cv2.bitwise_and(original_img, original_img, mask=mask)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(roi_gray, CANNY_LOWER, CANNY_UPPER)
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_mask_pixels if total_mask_pixels > 0 else 0

            if edge_density > EDGE_DENSITY_THRESHOLD and aspect_ratio <= ASPECT_RATIO_THRESHOLD:
                filtered_contours.append(cnt)
                filtered_edge_pixels.append(edge_pixels)

        # --- Clustering ---
        centroids = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        grouped_indices = []
        processed = [False] * len(filtered_contours)
        for i in range(len(filtered_contours)):
            if processed[i]:
                continue
            group = [i]
            processed[i] = True
            for j in range(i + 1, len(filtered_contours)):
                if not processed[j]:
                    dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)
                    if dist < MAX_DISTANCE:
                        group.append(j)
                        processed[j] = True
            if group:
                grouped_indices.append(group)

        # --- Visualization & Analysis ---
        highlighted_img = original_img.copy()
        img_with_obb = original_img.copy()
        
        total_circular = 0
        cluster_data = []
        labels_to_draw = [] 
        obb_data = []

        for idx, group_indices in enumerate(grouped_indices):
            cluster_id = idx + 1
            group_contours = [filtered_contours[k] for k in group_indices]
            total_edge_pixels = sum(filtered_edge_pixels[k] for k in group_indices)

            # --- Size Classification ---
            if total_edge_pixels < SMALL_EDGE_THRESHOLD:
                size_cat = "Small"
            elif total_edge_pixels < MEDIUM_EDGE_THRESHOLD:
                size_cat = "Medium"
            else:
                size_cat = "Large"

            # --- Get color from SIZE_COLORS ---
            color = SIZE_COLORS[size_cat]

            # --- Semi-transparent fill (Higher opacity for contrast) ---
            overlay = highlighted_img.copy()
            cv2.drawContours(overlay, group_contours, -1, color, thickness=cv2.FILLED)
            alpha = 0.35  # Increased from 0.4 → stronger color
            cv2.addWeighted(overlay, alpha, highlighted_img, 1 - alpha, 0, highlighted_img)

            # --- OBB (same color) ---
            all_points = np.concatenate(group_contours, axis=0)
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_with_obb, [box], 0, color, 3)  # Thicker line
            obb_data.append((box, color))

            # --- Circular buds (Bright Yellow) ---
            circular_count = 0
            for cnt in group_contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if circularity > CIRCULARITY_THRESHOLD:
                        circular_count += 1
                        total_circular += 1
                        cv2.drawContours(highlighted_img, [cnt], -1, (0, 255, 255), thickness=cv2.FILLED)

            # --- Label: ID + Size (High contrast text) ---
            center = (int(rect[0][0]), int(rect[0][1]))
            label = f"ID:{cluster_id} {size_cat}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8  # Larger text
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2

            final_img_f = out_dir / f"{stem}_FINAL_VISUALIZATION_F.png"
            cv2.imwrite(str(final_img_f), highlighted_img)
            labels_to_draw.append((label, text_x, text_y, text_size, font, font_scale, thickness))
            
            # Save cluster data
            cluster_data.append({
                "cluster_id": cluster_id,
                "size": size_cat,
                "edge_pixels": total_edge_pixels,
                "num_contours": len(group_indices),
                "num_circular": circular_count,
                "centroid": center
            })

            # Write to CSV
            csv_writer.writerow([
                filled_mask_path.name, cluster_id, size_cat, total_edge_pixels,
                len(group_indices), circular_count, center[0], center[1]
            ])
            
        highlighted_img_id = highlighted_img.copy()
        combined_img = highlighted_img.copy()

        # Draw OBBs on combined_img
        for box, color in obb_data:
            cv2.drawContours(combined_img, [box], 0, color, 3)

        padding = 8
        for (label, text_x, text_y, text_size, font, font_scale, thickness) in labels_to_draw:
            # Draw on highlighted_img_id
            cv2.rectangle(
                highlighted_img_id,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1)
            cv2.putText(
                highlighted_img_id, label, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)
            
            # Draw on combined_img
            cv2.rectangle(
                combined_img,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1)
            cv2.putText(
                combined_img, label, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)

        # --- Save Outputs ---
        final_vis_out = out_dir / f"{stem}_FINAL_VISUALIZATION.png"
        obb_out = out_dir / f"{stem}_CLUSTER_OBB.png"
        combined_out = out_dir / f"{stem}_COMBINED_VISUALIZATION.png"
        
        cv2.imwrite(str(final_vis_out), highlighted_img_id)
        cv2.imwrite(str(obb_out), img_with_obb)
        cv2.imwrite(str(combined_out), combined_img)

        # --- Save Annotations ---
        annot_path = annot_dir / f"{stem}.txt"
        h_img, w_img = original_img.shape[:2]
        with open(annot_path, "w") as f_annot:
            for box, _ in obb_data:
                # box is np.int0 of 4 points. Get boundingRect of these points for AABB
                x, y, w, h = cv2.boundingRect(box)
                
                # Normalize
                cx = (x + w / 2.0) / w_img
                cy = (y + h / 2.0) / h_img
                nw = w / w_img
                nh = h / h_img
                
                # Clamp values to [0, 1] just in case
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                f_annot.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        print(f"     -> Annotations: {annot_path.name}")

        # Per-image stats
        size_counts = Counter(c["size"] for c in cluster_data)
        summary_stats.append({
            "image": filled_mask_path.name,
            "total_clusters": len(grouped_indices),
            "small": size_counts["Small"],
            "medium": size_counts["Medium"],
            "large": size_counts["Large"],
            "total_circular": total_circular
        })

        print(f"[ok] {filled_mask_path.name} → {len(grouped_indices)} clusters "
              f"(S:{size_counts['Small']} M:{size_counts['Medium']} L:{size_counts['Large']}) "
              f"→ {final_vis_out.name}")

# Close CSV
csv_file.close()

# === FINAL SUMMARY REPORT ===
report_path = out_dir / "FINAL_CLUSTER_SUMMARY_REPORT.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("APPLE BLOSSOM CLUSTER ANALYSIS - FINAL REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total images processed: {len(summary_stats)}\n\n")
    f.write("Per-Image Summary:\n")
    f.write("-" * 70 + "\n")

    total_clusters = total_circular_global = 0
    size_totals = {"Small": 0, "Medium": 0, "Large": 0}

    for stat in summary_stats:
        f.write(f"{stat['image']}\n")
        f.write(f"  Total Clusters: {stat['total_clusters']}\n")
        f.write(f"    Small  : {stat['small']}  (Yellow)\n")
        f.write(f"    Medium : {stat['medium']} (Green)\n")
        f.write(f"    Large  : {stat['large']}  (Orange)\n")
        f.write(f"  Circular buds: {stat['total_circular']}\n\n")

        total_clusters += stat['total_clusters']
        total_circular_global += stat['total_circular']
        size_totals["Small"] += stat["small"]
        size_totals["Medium"] += stat["medium"]
        size_totals["Large"] += stat["large"]

    f.write("="*70 + "\n")
    f.write("GLOBAL SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total clusters detected: {total_clusters}\n")
    f.write(f"  Small  (< {SMALL_EDGE_THRESHOLD} edge px): {size_totals['Small']}  (Yellow)\n")
    f.write(f"  Medium ({SMALL_EDGE_THRESHOLD}–{MEDIUM_EDGE_THRESHOLD} edge px): {size_totals['Medium']} (Green)\n")
    f.write(f"  Large  (> {MEDIUM_EDGE_THRESHOLD} edge px): {size_totals['Large']}  (Orange)\n")
    f.write(f"Total circular buds: {total_circular_global}\n")

print(f"\nDetailed CSV saved: {csv_path}")
print(f"Final report saved: {report_path}")