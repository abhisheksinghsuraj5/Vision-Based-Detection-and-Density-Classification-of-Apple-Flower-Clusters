import os
import numpy as np
import cv2
import torch, time, platform
from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration ---
INPUT_FOLDER = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1_depth_anything\Image"  # Specify your input folder here
OUTPUT_FOLDER = r"H:\ITL\Methodology_2\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\final_steps\step1_depth_anything\output"  # Specify your output folder here
BACKGROUND_PROCESSING = "black"  # Fixed to "black" as requested

if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Input folder not found at '{INPUT_FOLDER}'")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output will be saved to: '{OUTPUT_FOLDER}'")

print("--- Step 1/X: Loading Depth Anything V2 Model ---")
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# If using CUDA, print the GPU name
if DEVICE == 'cuda':
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA GPU: {device_name}")
elif DEVICE == 'mps':
    print("MPS device: Apple Silicon GPU (specific GPU name not accessible via PyTorch)")
elif DEVICE == 'cpu':
    device_name = platform.processor() or "Unknown CPU"
    print(f"CPU: {device_name}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb'
model_path = "E:\project arbeit 1\Depth-Anything-V2-flower-detection\Depth-Anything-V2-flower-detection\checkpoints\depth_anything_v2_vitb.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'")

# Load Depth Anything V2 model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.to(DEVICE).eval()
print(f"Depth model '{encoder}' loaded successfully.")

# --- Process Images in Folder ---
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

if not image_files:
    print(f"No image files found in '{INPUT_FOLDER}'. Exiting.")
    exit()

for i, image_filename in enumerate(image_files):
    image_path = os.path.join(INPUT_FOLDER, image_filename)
    base_filename, ext = os.path.splitext(image_filename)

    print(f"\n--- Processing Image {i + 1}/{len(image_files)}: '{image_filename}' ---")

    # --- Load Image and Generate Foreground ---
    print("--- Step 1.1: Generating Foreground Image with Depth ---")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Warning: Could not read image '{image_path}'. Skipping.")
        continue

    # Run depth inference (returns HxW numpy depth map, lower values = closer)
    start_time = time.time()
    depth = model.infer_image(raw_img)
    end_time = time.time()
    depth_processing_time = end_time - start_time
    print(f"Depth Anything V2 inference time: {depth_processing_time:.4f} seconds in {DEVICE} - {device_name}")

    # Normalize depth map for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    #cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{base_filename}_depth_visualization.png'), depth_colored)
    print(f"Saved colored depth map to '{os.path.join(OUTPUT_FOLDER, f'{base_filename}_depth_visualization.png')}'")

    # Create foreground mask (nearby objects = 255, background = 0)
    THRESHOLD_PERCENTAGE = 0.3  # Adjust (0.2 for closer objects, 0.5 for more objects)
    threshold_value = depth.min() + (depth.max() - depth.min()) * THRESHOLD_PERCENTAGE
    foreground_mask = (depth > threshold_value).astype(np.uint8) * 255

    # Refine mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{base_filename}_foreground_mask.png'), foreground_mask)
    print(f"Saved binary mask to '{os.path.join(OUTPUT_FOLDER, f'{base_filename}_foreground_mask.png')}'")

    # Apply background processing (fixed to black)
    foreground_img = raw_img.copy()
    print("--- Step 1.2: Setting background to black ---")
    foreground_img[foreground_mask == 0] = [0, 0, 0]  # Set background pixels to black (BGR)
    output_filename = os.path.join(OUTPUT_FOLDER, f'{base_filename}.png')

    cv2.imwrite(output_filename, foreground_img)
    print(f"Saved foreground image with black background to '{output_filename}'")

print("\n--- All images processed. ---")