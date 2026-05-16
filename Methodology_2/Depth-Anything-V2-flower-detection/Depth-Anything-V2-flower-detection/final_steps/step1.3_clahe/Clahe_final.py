import os
import cv2
import numpy as np
import pandas as pd

# ---------- Folders ----------
# Set this to the folder containing the already-processed inputs (the ones this script reads & enhances)
input_dir  = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1_depth_anything\output"

# Output folder for enhanced results and the Excel log
output_dir = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1.3_clahe\output"
os.makedirs(output_dir, exist_ok=True)

# ---------- Settings ----------
BLACK_THRESH = 8              # threshold to detect "black" pixels (ignore these when measuring brightness)
BRIGHTNESS_THRESHOLD = 63     # if mean foreground brightness < this → apply CLAHE
CLAHE_TILE_SIZE = (6, 6)
CLAHE_CLIP = 3.0

# Excel log file
excel_path = os.path.join(output_dir, "brightness_processing_log.xlsx")

results = []


# ---------- Helper Functions ----------
def apply_clahe(img, clip=2.0, tiles=(8, 8)):
    """
    Apply CLAHE to the L channel in LAB space; return BGR image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------- Process All Images ----------
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(valid_exts):
        continue

    fpath_processed = os.path.join(input_dir, fname)

    img_processed = cv2.imread(fpath_processed)
    if img_processed is None:
        print(f" Skipping unreadable processed file: {fname}")
        continue

    # --- Calculate foreground brightness (ignore black background) on the processed input ---
    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    mask_foreground = gray > BLACK_THRESH
    mean_brightness = float(np.mean(gray[mask_foreground])) if np.any(mask_foreground) else 0.0

    # --- Apply enhancement based on brightness ---
    if mean_brightness < BRIGHTNESS_THRESHOLD:
        enhanced = apply_clahe(img_processed, clip=CLAHE_CLIP, tiles=CLAHE_TILE_SIZE)
        process_type = "CLAHE only"
    else:
        enhanced = img_processed
        process_type = "No processing"

    # --- No merging with original; just use the enhanced image as final ---
    final = enhanced

    # --- Save processed image ---
    out_path = os.path.join(output_dir, fname)
    ok = cv2.imwrite(out_path, final)
    if not ok:
        print(f" Failed to write output for: {fname}")

    # --- Log result ---
    results.append({
        "image_name": fname,
        "mean_foreground_brightness": round(mean_brightness, 3),
        "processing": process_type,
    })

    print(f"{fname}  {process_type} (mean={mean_brightness:.2f})")

# ---------- Save results to Excel ----------
df = pd.DataFrame(results)
df.to_excel(excel_path, index=False)
print(f"\n Processing complete! Log saved to:\n{excel_path}")
