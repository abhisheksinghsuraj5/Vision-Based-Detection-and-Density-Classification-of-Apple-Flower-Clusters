# hsv_detect_blossoms_flat.py
import cv2
import numpy as np
from pathlib import Path

# === Configure folders ===
INPUT_DIR  = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1_depth_anything\output"
OUTPUT_DIR = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step2\output1"

# === HSV thresholds ===
# White / near-white blossoms
LOWER_WHITE = np.array([0, 0, 100], dtype=np.uint8)
UPPER_WHITE = np.array([180, 80, 255], dtype=np.uint8)

# Brightness gate for alternative thresholds
BRIGHTNESS_THRESHOLD = 63  # mean brightness threshold (ignoring black background)
BLACK_THRESH = 10           # gray-level below which pixels are treated as background

# Light pink blossoms (upper hue range)
LOWER_PINK  = np.array([160, 10, 150], dtype=np.uint8)
UPPER_PINK  = np.array([180, 100, 255], dtype=np.uint8)

# White / near-white blossoms (alternative set for darker images)
LOWER_WHITE1 = np.array([0, 0, 10], dtype=np.uint8)
UPPER_WHITE1 = np.array([18, 80, 255], dtype=np.uint8)

# Light pink blossoms (alternative set for darker images)
LOWER_PINK1  = np.array([160, 10, 15], dtype=np.uint8)
UPPER_PINK1  = np.array([18, 100, 255], dtype=np.uint8)

# === Morphology params ===
KERNEL_SIZE = 3
OPEN_ITERS  = 1
CLOSE_ITERS = 1

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images_recursive(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def process_image(in_path: Path, out_dir: Path):
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[warn] could not read: {in_path}")
        return

    img_processed = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

    # --- Calculate foreground brightness (ignore black background) on the processed input ---
    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    mask_foreground = gray > BLACK_THRESH
    mean_brightness = float(np.mean(gray[mask_foreground])) if np.any(mask_foreground) else 0.0

    # --- Choose thresholds based on brightness ---
    hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
    if mean_brightness < BRIGHTNESS_THRESHOLD:
        mask_white1 = cv2.inRange(hsv, LOWER_WHITE1, UPPER_WHITE1)
        mask_pink1  = cv2.inRange(hsv, LOWER_PINK1,  UPPER_PINK1)
        flower_mask_raw = cv2.bitwise_or(mask_white1, mask_pink1)
    else:
        mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        mask_pink  = cv2.inRange(hsv, LOWER_PINK,  UPPER_PINK)
        flower_mask_raw = cv2.bitwise_or(mask_white, mask_pink)

    # --- Morphology: open then close ---
    opened = cv2.morphologyEx(flower_mask_raw, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
    flower_mask_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

    # --- Save outputs ---
    stem = in_path.stem
    closed_out = out_dir / f"{stem}.png"
    cv2.imwrite(str(closed_out), flower_mask_closed)

    pct = (int(np.count_nonzero(flower_mask_closed)) / flower_mask_closed.size) * 100.0
    print(f"[ok] {in_path.name}: mean brightness {mean_brightness:.1f} -> {pct:.2f}% pixels in mask")


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    imgs = list(list_images_recursive(in_dir))
    if not imgs:
        print(f"[warn] no images found under: {in_dir}")
        return

    print(f"[info] found {len(imgs)} images. Writing masks to: {out_dir}")
    for i, p in enumerate(imgs, 1):
        try:
            process_image(p, out_dir)
        except Exception as e:
            print(f"[error] {p.name}: {e}")


if __name__ == "__main__":
    main()
