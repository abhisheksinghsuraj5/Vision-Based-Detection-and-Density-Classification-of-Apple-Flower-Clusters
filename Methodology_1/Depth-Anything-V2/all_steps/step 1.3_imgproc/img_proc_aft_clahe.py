import cv2
import numpy as np
import os

# ----- Paths (your folders) -----
input_folder  = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step1_depth anything\output"
output_folder = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step 1.3_imgproc\output"
os.makedirs(output_folder, exist_ok=True)

# ----- Tunables (start with these; tweak if needed) -----
bg_threshold   = 10    # separates black background from subject
erode_px       = 1     # trims contaminated rim (1–3 is typical)
feather_ksize  = 3     # 3,5,7; higher = softer edge
edge_band_px   = 3     # width of the band to re-color near edges

def safe_kernel(k):
    # odd and >= 3 for Gaussian blur
    return k if (k % 2 == 1 and k >= 3) else 5

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tif', '.bmp')):
        continue

    src_path = os.path.join(input_folder, fname)
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Skipping {fname} (cannot read).")
        continue

    h, w = img.shape[:2]

    # 1) Build initial mask from black background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY)

    # 2) Edge trim: erode (cuts off 1–2px halo), then small close to heal tiny holes
    erode_k = np.ones((erode_px*2+1, erode_px*2+1), np.uint8)
    mask_eroded = cv2.erode(mask, erode_k, iterations=1)
    mask_closed = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # 3) Make/identify the narrow edge band for inpainting
    band_outer = cv2.dilate(mask_closed, np.ones((edge_band_px*2+1, edge_band_px*2+1), np.uint8))
    edge_band = cv2.subtract(band_outer, mask_closed)  # band around the object

    # 4) Inpaint the rim using only foreground colors
    #    We want to replace contaminated edge colors with colors from just inside the object.
    #    Create an inpaint mask that targets ONLY the edge_band pixels inside the subject silhouette.
    band_inside = cv2.bitwise_and(edge_band, mask)  # restrict to near-edge FG
    # To guide inpainting from inner FG, temporarily "remove" the band pixels
    img_for_inpaint = img.copy()
    # Mark band pixels as unknown (set to 0); Telea will pull colors from neighbors (inner FG)
    img_for_inpaint[band_inside > 0] = 0
    cleaned = cv2.inpaint(img_for_inpaint, band_inside, 3, cv2.INPAINT_TELEA)

    # 5) Feather the (trimmed) mask for a clean transition
    mask_feather = cv2.GaussianBlur(mask_closed, (safe_kernel(feather_ksize), safe_kernel(feather_ksize)), 0)

    # 6) Alpha-premultiply (suppresses residual bright fringe contribution)
    alpha = (mask_feather.astype(np.float32) / 255.0)[..., None]  # HxWx1
    premultiplied = (cleaned.astype(np.float32) * alpha).astype(np.uint8)

    # 7) Optional: protect bright flowers (keep them at full alpha)
    #    If your petals are truly white, this keeps them from dimming in step 6.
    flower_protect = (img[:,:,0] > 220) & (img[:,:,1] > 220) & (img[:,:,2] > 220)
    alpha_fp = alpha.copy()
    alpha_fp[flower_protect] = 1.0
    premultiplied = (cleaned.astype(np.float32) * alpha_fp).astype(np.uint8)

    out_path = os.path.join(output_folder, fname)
    cv2.imwrite(out_path, premultiplied)
    print(f" Saved {out_path}")

print("Done.")
