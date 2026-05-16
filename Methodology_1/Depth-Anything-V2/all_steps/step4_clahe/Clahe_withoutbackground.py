import os
import cv2
import numpy as np
import pandas as pd

# ---------- Folders ----------
input_dir  = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step 1.3_imgproc\output"
output_dir = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step4_clahe\output"
os.makedirs(output_dir, exist_ok=True)

# ---------- Settings ----------
BLACK_THRESH = 8              # threshold to ignore black background
BRIGHTNESS_THRESHOLD = 63     # if below this → apply CLAHE (else do nothing)
CLAHE_TILE_SIZE = (6, 6)
CLAHE_CLIP = 3.0

# Excel log file
excel_path = os.path.join(output_dir, "brightness_processing_log.xlsx")

results = []

# ---------- Helper Functions ----------
def apply_clahe(img, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ---------- Process All Images ----------
for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        continue

    fpath = os.path.join(input_dir, fname)
    img = cv2.imread(fpath)
    if img is None:
        print(f" Skipping unreadable file: {fname}")
        continue

    # --- Calculate foreground brightness (ignore black background) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > BLACK_THRESH

    if np.any(mask):
        mean_brightness = np.mean(gray[mask])
    else:
        mean_brightness = 0.0

    # --- Apply enhancement based on brightness ---
    if mean_brightness < BRIGHTNESS_THRESHOLD:
        # CASE: Too dark → Only CLAHE
        final = apply_clahe(img, clip=CLAHE_CLIP, tiles=CLAHE_TILE_SIZE)
        process_type = "CLAHE only"
    else:
        # CASE: Bright enough → Do nothing
        final = img
        process_type = "No processing"

    # --- Save processed image ---
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, final)

    # --- Log result ---
    results.append({
        "image_name": fname,
        "mean_foreground_brightness": round(float(mean_brightness), 3),
        "processing": process_type
    })

    print(f"{fname}  {process_type} (mean={mean_brightness:.2f})")

# ---------- Save results to Excel ----------
df = pd.DataFrame(results)
df.to_excel(excel_path, index=False)
print(f"\n Processing complete! Log saved to:\n{excel_path}")
