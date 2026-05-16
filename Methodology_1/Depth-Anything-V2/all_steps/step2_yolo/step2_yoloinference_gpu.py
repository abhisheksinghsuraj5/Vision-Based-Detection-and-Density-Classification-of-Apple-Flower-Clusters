import os
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import time

warnings.filterwarnings("ignore")


class SimpleFlowerAnnotator:
  

    def __init__(
        self,
        model_path,
        confidence_threshold=0.0001,
        iou_threshold=0.45,
        use_resizing=False,
        target_size=(640, 640),
        min_polygon_area=150,
    ):
        # Choose device ONCE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        if self.device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_resizing = use_resizing
        self.target_size = target_size if use_resizing else None
        self.min_polygon_area = min_polygon_area
        

    # ---------- Load images ----------
    def get_unique_images(self, folder_path):
        folder = Path(folder_path)
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        all_files = []
        for ext in extensions:
            all_files.extend(folder.glob(f"*{ext}"))
            all_files.extend(folder.glob(f"*{ext.upper()}"))
        seen = set()
        out = []
        for f in sorted(all_files):
            base = f.stem.lower()
            if base not in seen:
                seen.add(base)
                out.append(f)
        return out

    # ---------- Mask → Polygon ----------
    def extract_mask_to_polygons(self, mask, min_area=150, thresh=0.4):
        """Convert a binary mask to polygon contours."""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask

        binary = (mask_np > thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            eps = 0.004 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) >= 3:
                pts = approx.reshape(-1, 2).astype(np.float32)
                h, w = mask_np.shape
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                polys.append(pts)
        return polys

    def scale_polygon(self, polygon, src_shape, dst_shape):
        """Scale polygon coordinates from src (h,w) → dst (h,w)."""
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape
        sx = dst_w / src_w
        sy = dst_h / src_h
        p = polygon.copy()
        p[:, 0] *= sx
        p[:, 1] *= sy
        p[:, 0] = np.clip(p[:, 0], 0, dst_w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, dst_h - 1)
        return p

    def polygon_to_yolo(self, polygon, img_w, img_h):
        """Convert polygon points to YOLO segmentation format."""
        if polygon is None or len(polygon) < 3:
            return None
        coords = []
        for x, y in polygon:
            nx = float(x) / img_w
            ny = float(y) / img_h
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            coords.extend([nx, ny])
        return coords if len(coords) >= 6 else None

    # ---------- Visualization ----------
    def draw_polygon_outlines(self, img_bgr, polygons):
        """Draw only polygon outlines (no fill)."""
        if not polygons:
            return img_bgr
        outline_color = (255, 255, 180)  # light yellow-cyan
        for poly in polygons:
            if poly is None or len(poly) < 3:
                continue
            pts = poly.astype(np.int32)
            cv2.polylines(img_bgr, [pts], True, outline_color, thickness= 1, lineType=cv2.LINE_AA)
        return img_bgr

    # ---------- Run ----------
    def run(self, images_folder, output_folder):
        ann_dir = os.path.join(output_folder, "annotations")
        vis_dir = os.path.join(output_folder, "outlines")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        if self.use_resizing:
            processed_folder = os.path.join(output_folder, "resized_images")
            os.makedirs(processed_folder, exist_ok=True)

        imgs = self.get_unique_images(images_folder)
        print(f" Found {len(imgs)} images")
        print(f"  Writing annotations to: {ann_dir}")
        print(f"  Writing outline overlays to: {vis_dir}")
        
        yolo_device = 0 if self.device.type == "cuda" else "cpu"
        use_half = True if self.device.type == "cuda" else False

        total_time = 0.0
        count = 0

        for img_path in imgs:
            try:
                img_pil = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img_pil.size

                if self.use_resizing:
                    resized = img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
                    proc_path = os.path.join(output_folder, "resized_images", Path(img_path).name)
                    resized.save(proc_path)
                    infer_path = proc_path
                    inf_w, inf_h = self.target_size
                else:
                    infer_path = str(img_path)
                    inf_w, inf_h = orig_w, orig_h

                # ---- YOLO inference with timing ----
                start_time = time.time()
                results = self.model(
                    infer_path,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    save=False,
                    verbose=False,
                )
                inference_time = time.time() - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0.0
                print(f" {Path(img_path).name}: {inference_time:.3f} sec | {fps:.2f} FPS")

                total_time += inference_time
                count += 1

                polygons_img_space = []
                yolo_lines = []

                if hasattr(results[0], "masks") and results[0].masks is not None:
                    masks = results[0].masks
                    boxes = results[0].boxes

                    for i in range(len(masks)):
                        cls_id = int(boxes.cls[i].detach().cpu().numpy()) if hasattr(boxes, "cls") else 0
                        mask = masks.data[i]
                        m_h, m_w = mask.shape
                        polys = self.extract_mask_to_polygons(mask, min_area=self.min_polygon_area)
                        for poly in polys:
                            poly_inf = self.scale_polygon(poly, (m_h, m_w), (inf_h, inf_w))
                            poly_orig = (
                                self.scale_polygon(poly_inf, (inf_h, inf_w), (orig_h, orig_w))
                                if self.use_resizing
                                else poly_inf
                            )
                            polygons_img_space.append(poly_orig)
                            coords = self.polygon_to_yolo(poly_orig, orig_w, orig_h)
                            if coords:
                                line = f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords)
                                yolo_lines.append(line)

                # save annotations
                ann_file = os.path.join(ann_dir, f"{Path(img_path).stem}.txt")
                with open(ann_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

                # save outline image (no fill)
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    vis = self.draw_polygon_outlines(img_bgr, polygons_img_space)
                    vis_path = os.path.join(vis_dir, f"{Path(img_path).stem}.jpg")
                    cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

                print(f" {Path(img_path).name}: {len(yolo_lines)} polygons\n")

            except Exception as e:
                print(f"  Error on {img_path}: {e}")
                ann_file = os.path.join(ann_dir, f"{Path(img_path).stem}.txt")
                with open(ann_file, "w", encoding="utf-8") as f:
                    pass

        # ---- Average inference stats ----
        if count > 0:
            avg_time = total_time / count
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
            print(f"\n Average inference time: {avg_time:.3f} sec/image | {avg_fps:.2f} FPS")




def main():
    MODEL_PATH = "yolo_seg_l.pt"

    #  Updated paths:
    IMAGES_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step 1.3_imgproc\output"
    OUTPUT_FOLDER = r"H:\ITL\Methodology_1\Depth-Anything-V2\all_steps\step2_yolo\output"

    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    USE_RESIZING = False
    TARGET_SIZE = (640, 640)
    MIN_POLY_AREA = 150

    print("=== SIMPLE FLOWER ANNOTATION (OUTLINE ONLY) ===")
    annotator = SimpleFlowerAnnotator(
        MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        use_resizing=USE_RESIZING,
        target_size=TARGET_SIZE,
        min_polygon_area=MIN_POLY_AREA,
    )
    annotator.run(IMAGES_FOLDER, OUTPUT_FOLDER)
    print("\n Done! Check:")
    print(f" - Annotations: {OUTPUT_FOLDER}\\annotations")
    print(f" - Outlines:    {OUTPUT_FOLDER}\\outlines")


if __name__ == "__main__":
    main()
