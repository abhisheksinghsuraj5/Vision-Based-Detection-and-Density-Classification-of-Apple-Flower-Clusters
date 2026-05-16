# step3_find_contours.py
import cv2
import numpy as np
from pathlib import Path
import csv

# === Folders ===
INPUT_DIR  = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step2\output1"
OUTPUT_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step3\output"
OUTPUT_DIR1 = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step3\output1"

# === Parameters ===
AREA_THRESHOLD = 10   # drop contours smaller than this many pixels
# Only process masks produced in step2 (closed masks)
MASK_SUFFIX = "_flower_mask_closed.png"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# --- setup ---
in_dir = Path(INPUT_DIR)
out_dir = Path(OUTPUT_DIR)
out_dir1 = Path(OUTPUT_DIR1)
out_dir.mkdir(parents=True, exist_ok=True)
out_dir1.mkdir(parents=True, exist_ok=True)

# CSV summary
csv_path = out_dir / "contour_summary.csv"
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["image_name", "total_contours", "kept_contours", "area_threshold_px"])

# Find candidate mask files
files = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in VALID_EXTS])


if not files:
    print(f"No mask files matching *{MASK_SUFFIX} under: {in_dir}")
else:
    for mask_path in files:
        # Read mask in grayscale (0/255)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[skip] Cannot read: {mask_path}")
            continue

        # --- find contours on the binary mask ---
        # Ensure mask is 0/255
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter tiny contours
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > AREA_THRESHOLD]

        # --- filled contour mask (white blobs = kept contours) ---
        filled = np.zeros_like(bin_mask)
        if filtered:
            cv2.drawContours(filled, filtered, -1, color=255, thickness=-1)

        # --- outline preview (on a gray background for visibility) ---
        outline = np.full_like(bin_mask, 30)              # dark gray background
        outline[bin_mask > 0] = 80                        # original mask area slightly lighter
        if filtered:
            cv2.drawContours(outline, filtered, -1, color=255, thickness=1)  # white outlines

        # --- save outputs ---
        stem = mask_path.stem.replace("_flower_mask_closed", "")
        filled_out  = out_dir / f"{stem}.png"
        outline_out = out_dir1 / f"{stem}_contours_outline.png"
        cv2.imwrite(str(filled_out), filled)
        cv2.imwrite(str(outline_out), outline)

        # Log to CSV
        csv_writer.writerow([mask_path.name, len(contours), len(filtered), AREA_THRESHOLD])

        print(f"[ok] {mask_path.name}: kept {len(filtered)}/{len(contours)} contours -> {filled_out.name}")

csv_file.close()
print(f"\nSummary saved to: {csv_path}")
