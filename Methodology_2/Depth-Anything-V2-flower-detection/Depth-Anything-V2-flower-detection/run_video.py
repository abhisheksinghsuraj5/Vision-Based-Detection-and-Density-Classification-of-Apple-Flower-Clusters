import argparse
import os
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from depth_anything_v2.dpt import DepthAnythingV2

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Perform panoptic segmentation on depth-based foreground, highlight 'white' flowers, and cluster them into 'bunches'.")
parser.add_argument(
    "image_path",
    type=str,
    help="Path to the input image file (e.g., image.jpg or image.png)"
)
args = parser.parse_args()

if not os.path.exists(args.image_path):
    print(f"Error: Image file not found at '{args.image_path}'")
    exit()

# --- Depth Anything V2 Setup ---
print("--- Step 1/6: Loading Depth Anything V2 Model ---")
DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

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

# --- Load Image and Generate Foreground ---
print("--- Step 2/6: Generating Foreground Image with Depth ---")
raw_img = cv2.imread(args.image_path)
if raw_img is None:
    raise FileNotFoundError(f"Image not found at '{args.image_path}'")

# Run depth inference (returns HxW numpy depth map, lower values = closer)
depth = model.infer_image(raw_img)

# Normalize depth map for visualization
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth_normalized = depth_normalized.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
cv2.imwrite('output_depth_visualization.png', depth_colored)
print("Saved colored depth map to 'output_depth_visualization.png'")

# Create foreground mask (nearby objects = 255, background = 0)
THRESHOLD_PERCENTAGE = 0.3  # Adjust (0.2 for closer objects, 0.5 for more objects)
threshold_value = depth.min() + (depth.max() - depth.min()) * THRESHOLD_PERCENTAGE
foreground_mask = (depth > threshold_value).astype(np.uint8) * 255

# Refine mask with morphological operations
kernel = np.ones((5, 5), np.uint8)
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('output_foreground_mask.png', foreground_mask)
print("Saved binary mask to 'output_foreground_mask.png'")

# Apply blur to the background and keep foreground unblurred
blurred_img = cv2.GaussianBlur(raw_img, (11, 11), 2)  # Apply Gaussian blur to the entire image
foreground_img = raw_img.copy()
foreground_img[foreground_mask == 0] = blurred_img[foreground_mask == 0]  # Replace background with blurred version
cv2.imwrite('output_foreground_image.png', foreground_img)
print("Saved foreground image with blurred background to 'output_foreground_image.png'")

# Convert foreground image to RGB for MaskFormer
foreground_img_rgb = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(foreground_img_rgb)
original_image_size = image.size[::-1]

# --- MaskFormer Model and Segmentation ---
print("--- Step 3/6: Loading MaskFormer Model ---")
processor = MaskFormerImageProcessor.from_pretrained(
    "facebook/maskformer-swin-base-coco",
    do_resize=True,
    do_rescale=True,
    do_normalize=True,
)
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

print("--- Step 4/6: Finding All Flowers in the Foreground Image ---")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[original_image_size])[0]
predicted_panoptic_map = result["segmentation"].cpu().numpy()

# --- Isolate 'Flower' Segments ---
segment_info_list = result["segments_info"]
flower_label_id = -1
for label_id, label_name in model.config.id2label.items():
    if label_name.lower() == "flower":
        flower_label_id = label_id
        break

if flower_label_id == -1:
    print("Warning: 'flower' class not found in model's labels. No flowers will be segmented.")
    flower_mask = np.zeros_like(predicted_panoptic_map, dtype=np.uint8)
else:
    flower_mask = np.zeros_like(predicted_panoptic_map, dtype=np.uint8)
    for segment in segment_info_list:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        if label_id == flower_label_id:
            flower_mask[predicted_panoptic_map == segment_id] = 255

# --- Filter for 'White' Pixels and Cluster ---
print("--- Step 5/6: Identifying White Flower Bunches ---")
hsv_image = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 100])
upper_white = np.array([180, 80, 255])
white_color_mask = cv2.inRange(hsv_image, lower_white, upper_white)
white_flower_pixels_mask = cv2.bitwise_and(flower_mask, white_color_mask)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_flower_pixels_mask, 8, cv2.CV_32S)
min_cluster_area = 120

bounding_boxes, cluster_indices = [], []
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_cluster_area:
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bounding_boxes.append([x, y, x + w, y + h])
        cluster_indices.append(i)

if len(bounding_boxes) > 0:
    bounding_boxes = np.array(bounding_boxes, dtype=np.float32)

    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1; x3, y3, x4, y4 = box2
        inter_x1, inter_y1 = max(x1, x3), max(y1, y3); inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area, box2_area = (x2 - x1) * (y2 - y1), (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def are_nearby(box1, box2, dist_threshold=60):
        if compute_iou(box1, box2) > 0: return True
        center1, center2 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2], [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2) < dist_threshold

    num_boxes = len(bounding_boxes)
    adj_matrix = np.zeros((num_boxes, num_boxes), dtype=bool)
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if are_nearby(bounding_boxes[i], bounding_boxes[j]): adj_matrix[i, j] = adj_matrix[j, i] = True

    visited, box_clusters = [False] * num_boxes, []
    for i in range(num_boxes):
        if not visited[i]:
            cluster, stack = [], [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True; cluster.append(node)
                    for j in range(num_boxes):
                        if adj_matrix[node, j] and not visited[j]: stack.append(j)
            box_clusters.append(cluster)

    merged_boxes = []
    for cluster in box_clusters:
        if cluster:
            cluster_boxes = bounding_boxes[cluster]
            xmin, ymin = np.min(cluster_boxes[:, 0]), np.min(cluster_boxes[:, 1])
            xmax, ymax = np.max(cluster_boxes[:, 2]), np.max(cluster_boxes[:, 3])
            merged_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    
    significant_bunches = len(merged_boxes)
else:
    box_clusters, merged_boxes, significant_bunches = [], [], 0

# --- Analysis and Visualization ---
print("--- Step 6/6: Analyzing Bunch Size and Density ---")
SIZE_THRESHOLD_LARGE = 6000
SIZE_THRESHOLD_MEDIUM = 3000
DENSITY_THRESHOLD_SPARSE = 0.4

COLOR_LARGE = (0, 255, 0); COLOR_MEDIUM = (0, 165, 255); COLOR_SMALL = (255, 255, 0)
BOX_COLOR = (0, 0, 255); TEXT_COLOR = (255, 255, 255)
DENSE_OUTLINE_COLOR = (255, 255, 255); SPARSE_OUTLINE_COLOR = (0, 0, 0)

analyzed_image = foreground_img.copy()
colored_bunches_mask = np.zeros_like(foreground_img)
analysis_summary = {'large': 0, 'medium': 0, 'small': 0, 'total_pixels': 0}

if significant_bunches > 0:
    white_flower_edges = cv2.Canny(white_flower_pixels_mask, 100, 200)
    cv2.imwrite("white_flower_edges.png", white_flower_edges)

    for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
        current_bunch_mask = np.zeros_like(labels, dtype=np.uint8)
        for component_idx in box_clusters[i]:
            current_bunch_mask[labels == cluster_indices[component_idx]] = 255
        
        cluster_size = np.count_nonzero(current_bunch_mask)
        if cluster_size == 0: continue
        
        analysis_summary['total_pixels'] += cluster_size
        edge_pixel_count = np.count_nonzero(cv2.bitwise_and(white_flower_edges, white_flower_edges, mask=current_bunch_mask))
        edge_density = edge_pixel_count / cluster_size

        if cluster_size > SIZE_THRESHOLD_LARGE:
            fill_color = COLOR_LARGE; size_label = "Large"; analysis_summary['large'] += 1
        elif cluster_size >= SIZE_THRESHOLD_MEDIUM:
            fill_color = COLOR_MEDIUM; size_label = "Medium"; analysis_summary['medium'] += 1
        else:
            fill_color = COLOR_SMALL; size_label = "Small"; analysis_summary['small'] += 1

        is_sparse = edge_density > DENSITY_THRESHOLD_SPARSE
        outline_color = SPARSE_OUTLINE_COLOR if is_sparse else DENSE_OUTLINE_COLOR

        colored_bunches_mask[current_bunch_mask == 255] = fill_color
        fill_overlay = np.zeros_like(analyzed_image)
        fill_overlay[current_bunch_mask == 255] = fill_color
        analyzed_image = cv2.addWeighted(analyzed_image, 1, fill_overlay, 0.6, 0)
        cv2.rectangle(analyzed_image, (x1, y1), (x2, y2), outline_color, 2)
        cv2.putText(analyzed_image, size_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

# --- Generate Final Report ---
def create_summary_report(original_img, analyzed_img, summary_data):
    h, w, _ = original_img.shape
    report_h, report_w = h + 150, w * 2 + 30
    report_canvas = np.full((report_h, report_w, 3), 20, np.uint8)
    report_canvas[0:h, 10:w+10] = original_img
    report_canvas[0:h, w+20:w*2+20] = analyzed_img
    cv2.putText(report_canvas, "Original Image", (10, h + 30), cv2.FONT_HERSHEY_DUPLEX, 1, TEXT_COLOR, 2)
    cv2.putText(report_canvas, "Analysis Result", (w + 20, h + 30), cv2.FONT_HERSHEY_DUPLEX, 1, TEXT_COLOR, 2)
    total_bunches = sum(summary_data.values()) - summary_data['total_pixels']
    cv2.putText(report_canvas, f"Total Bunches Found: {total_bunches}", (10, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
    cv2.putText(report_canvas, f" - Large (> {SIZE_THRESHOLD_LARGE}px): {summary_data['large']}", (10, h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LARGE, 2)
    cv2.putText(report_canvas, f" - Medium ({SIZE_THRESHOLD_MEDIUM}-{SIZE_THRESHOLD_LARGE}px): {summary_data['medium']}", (450, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_MEDIUM, 2)
    cv2.putText(report_canvas, f" - Small (< {SIZE_THRESHOLD_MEDIUM}px): {summary_data['small']}", (450, h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SMALL, 2)
    cv2.rectangle(report_canvas, (w*2-220, h+60), (w*2-200, h+80), DENSE_OUTLINE_COLOR, -1)
    cv2.putText(report_canvas, "Dense Bunch", (w*2-190, h+78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    cv2.rectangle(report_canvas, (w*2-220, h+90), (w*2-200, h+110), SPARSE_OUTLINE_COLOR, -1)
    cv2.putText(report_canvas, "Sparse Bunch", (w*2-190, h+108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    return report_canvas

final_report = create_summary_report(foreground_img, analyzed_image, analysis_summary)

# --- Save and Display ---
report_filename = os.path.splitext(args.image_path)[0] + "_summary_report.jpg"
detailed_filename = os.path.splitext(args.image_path)[0] + "_detailed_analysis.jpg"
colored_mask_filename = os.path.splitext(args.image_path)[0] + "_bunches_colored_mask.jpg"

cv2.imwrite(report_filename, final_report)
cv2.imwrite(detailed_filename, analyzed_image)
cv2.imwrite(colored_mask_filename, colored_bunches_mask)

print("\n--- ANALYSIS COMPLETE ---")
print(f"Found {sum(analysis_summary.values()) - analysis_summary['total_pixels']} white flower bunches.")
print(f"  - Large:   {analysis_summary['large']}")
print(f"  - Medium:  {analysis_summary['medium']}")
print(f"  - Small:   {analysis_summary['small']}")
print("\nRESULTS SAVED:")
print(f"  - Main Report:     {report_filename}")
print(f"  - Detailed View:   {detailed_filename}")
print(f"  - Color Mask Only: {colored_mask_filename}")
cv2.imshow("Flower Analysis Report", final_report)
cv2.imshow("Colored Bunches Mask Only", colored_bunches_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()