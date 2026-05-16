# hsv_detect_blossoms_flat.py
import cv2
import numpy as np
from pathlib import Path

# === Configure folders ===
INPUT_DIR  = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1.3_clahe\output"
OUTPUT_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step2\output"

# === HSV thresholds ===
# White / near-white blossoms
LOWER_WHITE = np.array([  0,   0, 100], dtype=np.uint8)
UPPER_WHITE = np.array([180,  80, 255], dtype=np.uint8)

# Light pink blossoms (upper hue range)
LOWER_PINK  = np.array([160,  10, 150], dtype=np.uint8)
UPPER_PINK  = np.array([180, 100, 255], dtype=np.uint8)

# White / near-white blossoms
LOWER_WHITE1 = np.array([  0,   0, 10], dtype=np.uint8)
UPPER_WHITE1 = np.array([18,  80, 255], dtype=np.uint8)

# Light pink blossoms (upper hue range)
LOWER_PINK1  = np.array([160,  10, 15], dtype=np.uint8)
UPPER_PINK1  = np.array([18, 100, 255], dtype=np.uint8)

# === Morphology params ===
KERNEL_SIZE = 3
OPEN_ITERS  = 1
CLOSE_ITERS = 1

# === File types to process ===
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# --- Setup ---
in_dir = Path(INPUT_DIR)
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS])
if not files:
    print(f"No images found under: {in_dir}")
else:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

    for in_path in files:
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] Could not read: {in_path}")
            continue

        # --- HSV color mask (white + pink) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        mask_pink  = cv2.inRange(hsv, LOWER_PINK,  UPPER_PINK)
        flower_mask_raw = cv2.bitwise_or(mask_white, mask_pink)

        # --- Morphology: open then close ---
        opened = cv2.morphologyEx(flower_mask_raw, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
        flower_mask_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

        # --- Save outputs ---
        stem = in_path.stem
        raw_out    = out_dir / f"{stem}_flower_mask_raw.png"
        closed_out = out_dir / f"{stem}.png"
        #cv2.imwrite(str(raw_out), flower_mask_raw)
        cv2.imwrite(str(closed_out), flower_mask_closed)

        pct = (int(np.count_nonzero(flower_mask_closed)) / flower_mask_closed.size) * 100.0
        print(f"[ok] {in_path.name}: {pct:.2f}% pixels in closed mask -> saved {closed_out.name}")
