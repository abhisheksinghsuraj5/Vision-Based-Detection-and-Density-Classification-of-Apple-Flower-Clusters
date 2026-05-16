import cv2
import torch
import numpy as np
import sys
import os
import time

# -----------------------------
# Helper: sync for accurate timing
# -----------------------------
def _sync(device: str):
    """Ensure all queued ops finish before timing."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

# -----------------------------
# Set project path (keep yours)
# -----------------------------
PROJECT_ROOT = r"H:\ITL\Methodology_1\Depth-Anything-V2"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------
# Device: CUDA (Windows)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("Using device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Model config
# -----------------------------
model_configs = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

encoder = "vitl"  # keep vits for MX350 (2GB VRAM)

# -----------------------------
# Load model
# -----------------------------
model = DepthAnythingV2(**model_configs[encoder])

weights_path = rf"H:\ITL\Methodology_1\Depth-Anything-V2\depth_anything_v2_{encoder}.pth"
state = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)

model = model.to(DEVICE).eval()

# -----------------------------
# GPU warmup (important)
# -----------------------------
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
with torch.no_grad():
    if DEVICE == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model.infer_image(dummy)
        _sync(DEVICE)
    else:
        _ = model.infer_image(dummy)

print("Warmup done.")

def process_image(image_path, output_path, model, device, threshold_quantile=0.7):
    """Process a single image with depth estimation and foreground extraction.
       Returns (success: bool, inference_time_ms: float, total_time_ms: float)
    """
    t0 = time.perf_counter()

    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error loading image: {image_path}")
        return False, None, None
    
    print(f"Processing: {os.path.basename(image_path)} - Shape: {raw_img.shape}")

    # --- inference timing ---
    _sync(device)
    t_infer_start = time.perf_counter()
    depth = model.infer_image(raw_img)
    _sync(device)
    t_infer_end = time.perf_counter()
    inference_time_ms = (t_infer_end - t_infer_start) * 1000.0

    # Normalize depth map
    depth_normalized = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Threshold based on quantile
    threshold = np.quantile(depth_normalized, threshold_quantile)

    # Create foreground mask and invert
    mask_foreground = depth_normalized <= threshold
    mask_foreground = ~mask_foreground

    # Apply mask to original image
    output_img = raw_img.copy()
    output_img[~mask_foreground] = [0, 0, 0]

    # Save result
    cv2.imwrite(output_path, output_img)

    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0

    print(f" Saved: {os.path.basename(output_path)} | Inference: {inference_time_ms:.2f} ms | Total: {total_time_ms:.2f} ms")
    return True, inference_time_ms, total_time_ms

    return True

def process_folder(input_folder, output_folder, model, device, threshold_quantile=0.7):
    """Process all images in a folder and report timing stats."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    processed_count = 0
    failed_count = 0
    infer_times = []
    total_times = []
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{filename}"
        output_path = os.path.join(output_folder, output_filename)
        
        success, t_infer_ms, t_total_ms = process_image(input_path, output_path, model, device, threshold_quantile)
        
        if success:
            processed_count += 1
            infer_times.append(t_infer_ms)
            total_times.append(t_total_ms)
        else:
            failed_count += 1
    
    print(f"\n=== Processing Complete ===")
    print(f" Successfully processed: {processed_count} images")
    print(f" Failed: {failed_count} images")
    print(f" Output folder: {output_folder}")

    if infer_times:
        avg_infer = sum(infer_times) / len(infer_times)
        avg_total = sum(total_times) / len(total_times)
        fps_infer = 1000.0 / avg_infer if avg_infer > 0 else float('inf')
        fps_total = 1000.0 / avg_total if avg_total > 0 else float('inf')
        print(f"\n--- Timing Stats ---")
        print(f" Avg Inference: {avg_infer:.2f} ms  (~{fps_infer:.2f} FPS, model only)")
        print(f" Avg Total:     {avg_total:.2f} ms  (~{fps_total:.2f} FPS, end-to-end)")


# Usage example:
if __name__ == "__main__":
    # Set your input and output folder paths
    input_folder = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step1_depth anything\Image"  # Change this path
    output_folder = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step1_depth anything\output"  # Change this path
    
    # Optional: adjust threshold (0.7 means 70th percentile)
    threshold_quantile = 0.5
    
    # Process the entire folder
    process_folder(input_folder, output_folder, model, DEVICE, threshold_quantile)
