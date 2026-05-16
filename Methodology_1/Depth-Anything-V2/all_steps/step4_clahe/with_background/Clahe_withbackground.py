import os
import cv2
import numpy as np
import pandas as pd

# ---------- Folders ----------
# Set this to the folder containing the ORIGINAL images
orig_input_dir = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step1_depth anything\Image"

# Set this to the folder containing the already-processed inputs (the ones this script reads & enhances)
input_dir      = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step 1.3_imgproc\output"

# Output folder for merged results and the Excel log
output_dir     = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step4_clahe\output"
os.makedirs(output_dir, exist_ok=True)

# ---------- Settings ----------
BLACK_THRESH = 8              # threshold to detect "black" pixels (all channels <= this are considered black)
BRIGHTNESS_THRESHOLD = 63     # if mean foreground brightness < this → apply CLAHE
CLAHE_TILE_SIZE = (6, 6)
CLAHE_CLIP = 3.0

# If original and processed image sizes differ, choose whether to auto-resize original to match processed
RESIZE_ORIGINAL_TO_MATCH = True
INTERP_FOR_RESIZE = cv2.INTER_LINEAR

# Excel log file
excel_path = os.path.join(output_dir, "brightness_processing_log.xlsx")

results = []

# ---------- Helper Functions ----------
def apply_clahe(img, clip=2.0, tiles=(8,8)):
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
    fpath_original  = os.path.join(orig_input_dir, fname)

    img_processed = cv2.imread(fpath_processed)
    if img_processed is None:
        print(f" Skipping unreadable processed file: {fname}")
        continue

    img_original = cv2.imread(fpath_original)
    if img_original is None:
        print(f" Skipping (missing original): {fname}")
        continue

    # --- Size alignment if needed ---
    if img_original.shape[:2] != img_processed.shape[:2]:
        if RESIZE_ORIGINAL_TO_MATCH:
            img_original = cv2.resize(
                img_original,
                (img_processed.shape[1], img_processed.shape[0]),
                interpolation=INTERP_FOR_RESIZE
            )
        else:
            print(f" Skipping (size mismatch, no resize): {fname}")
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

    # --- Merge: replace black pixels in enhanced with original pixels ---
    # A pixel is considered "black" if ALL 3 channels <= BLACK_THRESH
    black_mask = np.all(enhanced <= BLACK_THRESH, axis=2)

    final = enhanced.copy()
    replaced_count = int(np.count_nonzero(black_mask))
    if replaced_count > 0:
        final[black_mask] = img_original[black_mask]

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
        "black_pixels_replaced": replaced_count
    })

    print(f"{fname}  {process_type} (mean={mean_brightness:.2f}, replaced_black_px={replaced_count})")

# ---------- Save results to Excel ----------
df = pd.DataFrame(results)
df.to_excel(excel_path, index=False)
print(f"\n Processing complete! Log saved to:\n{excel_path}")
