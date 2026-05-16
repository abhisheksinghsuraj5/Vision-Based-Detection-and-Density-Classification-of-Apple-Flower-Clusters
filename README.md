# Vision-Based Detection and Density Classification of Apple Flower Clusters

## Overview

This project focuses on detecting apple flower clusters from RGB orchard images and classifying the detected clusters into density levels. The goal is to support automated flower monitoring, yield prediction, thinning decisions, and data-driven orchard management using computer vision and AI-based image analysis.

The work evaluates two different methodologies for apple flower cluster detection and density classification, combining depth estimation, segmentation, image enhancement, thresholding, edge detection, clustering, and statistical validation.

## Motivation

Apple flowering is often spatially inconsistent, which makes yield prediction and thinning decisions difficult. Manual inspection is time-consuming and subjective, especially in large orchards.

This project addresses the need for:

- Automated flower and flower-cluster detection
- Cluster-based flower density estimation
- Real-time or near-real-time orchard monitoring
- Improved decision support for thinning and yield prediction
- Robust processing under variable lighting and image-resolution conditions

## Problem Statement

The project aims to:

1. Detect apple flower clusters from RGB orchard images.
2. Classify detected clusters into density levels such as high, medium, low, and very low.
3. Reduce the impact of lighting variation and limited image resolution on detection and classification performance.

## Dataset

The YOLO-based methodology used a dataset of **339 images**:

| Split | Number of Images |
|---|---:|
| Training | 237 |
| Validation | 68 |
| Testing | 34 |

For validation, manual flower-dot annotations were created using LabelMe and compared against detected cluster regions.

## Methodology 1: Depth + YOLO Segmentation + Edge-Based Classification

Methodology 1 combines learned segmentation with image-processing-based density estimation.

### Pipeline

1. **Depth Estimation**
   - Uses Depth Anything V2 to generate depth maps from RGB images.
   - Depth information supports flower-cluster separation and spatial understanding.

2. **YOLO V11 Segmentation**
   - Uses YOLO V11 segmentation models to detect flower regions.
   - Multiple YOLO V11 variants were compared.

3. **Clustering and Polygon Merging**
   - Nearby segmentation polygons are grouped using DBSCAN.
   - Overlapping or touching polygons are merged while avoiding oversized cluster regions.

4. **Brightness Analysis and CLAHE**
   - Image brightness is analyzed.
   - CLAHE is applied conditionally to improve local contrast in poor lighting conditions.

5. **Thresholding and Edge Detection**
   - Flower pixels are detected inside each polygon.
   - Canny edge detection is used to extract flower edge pixels.

6. **Density Classification**
   - Cluster density is classified using edge-pixel counts.

### Density Classes

| Edge Pixels | Density Class |
|---:|---|
| < 50 | Very Low |
| 50–499 | Low |
| 500–1099 | Medium |
| ≥ 1100 | High |

### YOLO V11 Performance Comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO V11n | 0.51972 | 0.50317 | 0.42948 | 0.17691 |
| YOLO V11s | 0.52572 | 0.48095 | 0.41367 | 0.18427 |
| YOLO V11m | 0.55278 | 0.48366 | 0.48970 | 0.22122 |
| YOLO V11l | 0.61194 | 0.59902 | 0.53629 | 0.28715 |

YOLO V11l achieved the best segmentation performance among the tested variants.

## Methodology 2: Depth + Thresholding + Contour-Based Classification

Methodology 2 avoids YOLO training and relies more heavily on classical image processing.

### Pipeline

1. **Depth Estimation**
   - Uses Depth Anything V2 for depth-map generation.

2. **Brightness Analysis and Cluster Thresholding**
   - Applies brightness-based processing and thresholding for white and pink flower regions.
   - Binary masks are generated and dilated to fill gaps.

3. **Anomaly Removal and Contour Detection**
   - Removes non-flower anomalies caused by similar RGB values.
   - Uses contour aspect ratio and edge-density thresholds.
   - Extracts object boundaries using `cv2.findContours()`.

4. **Clustering and Density Classification**
   - Flower contours are clustered based on centroid proximity.
   - Clusters are classified using total edge-pixel counts.

### Density Classes

| Edge Pixels | Density Class |
|---:|---|
| < 500 | Low |
| 500–1100 | Medium |
| ≥ 1100 | High |

## Validation

Validation was performed using Pearson correlation and linear regression analysis between detected edge pixels and manually annotated flower counts.

### Methodology 1 Validation

- 25 images analyzed
- 3,491 manually annotated flower points
- 3,256 points located inside detected cluster polygons
- 235 points outside polygon boundaries
- 1,263 polygon-level observations

#### Correlation Results

| Classification | Correlation | Strength |
|---|---:|---|
| High | 0.815 | Very Strong Positive |
| Low | 0.702 | Strong Positive |
| Medium | 0.432 | Moderate |
| Very Low | 0.229 | Weak |

#### Linear Regression Results

| Classification | Total | Following | Outliers | % Following | Slope |
|---|---:|---:|---:|---:|---:|
| High | 70 | 68 | 2 | 97.1 | 0.005574 |
| Low | 839 | 796 | 43 | 94.9 | 0.005574 |
| Medium | 170 | 161 | 9 | 94.7 | 0.005574 |
| Very Low | 184 | 182 | 2 | 98.9 | 0.005574 |

## Methodology 2 Validation

- 25 images analyzed
- 1,057 polygon-level observations

### Correlation Results

| Classification | Correlation | Strength |
|---|---:|---|
| Large | 0.658 | Strong Positive |
| Medium | 0.638 | Strong Positive |
| Small | 0.473 | Moderate |

### Linear Regression Results

| Classification | Total | Following | Outliers | % Following | Slope |
|---|---:|---:|---:|---:|---:|
| Large | 115 | 109 | 6 | 94.8 | 0.004507 |
| Medium | 407 | 384 | 23 | 94.3 | 0.004507 |
| Small | 535 | 504 | 31 | 94.2 | 0.004507 |

## Comparative Analysis

| Criterion | Methodology 1 | Methodology 2 |
|---|---|---|
| Detection Robustness | More robust due to higher overall correlation | Less robust due to lower overall correlation |
| Illumination Sensitivity | Less sensitive because YOLO is used for detection | More sensitive because detection relies mainly on colour thresholding |
| Computational Efficiency | Lower efficiency due to Depth Anything + YOLO | Higher efficiency because no YOLO inference is required |
| Implementation Complexity | More complex because YOLO training is required | Simpler because no training is required |

## Conclusion

The project successfully detected apple flower clusters from RGB orchard images and classified them into density categories.

Key conclusions:

- Methodology 1 provides better detection robustness.
- YOLO-based segmentation reduces sensitivity to lighting variation.
- Edge-pixel counts show a meaningful relationship with manually annotated flower counts.
- Methodology 2 is simpler and computationally lighter but more sensitive to illumination and colour variation.
- The proposed approach can support orchard monitoring, yield estimation, and thinning-related decision-making.

## Future Work

Possible future improvements include:

- Validating Depth Anything V2 depth accuracy against reliable depth references such as LiDAR.
- Benchmarking alternative models for depth estimation and cluster detection.
- Improving the transition from edge-pixel-based density estimation to explicit flower counting.
- Expanding the dataset across different orchard conditions, lighting environments, and flowering stages.
- Testing real-time deployment on edge devices or mobile platforms.

## Technologies and Methods Used

- Python
- OpenCV
- YOLO V11 segmentation
- Depth Anything V2
- CLAHE image enhancement
- Thresholding
- Canny edge detection
- DBSCAN clustering
- Contour detection
- Pearson correlation analysis
- Linear regression validation
- LabelMe manual annotation

## Author

**Abhishek Singh Suraj**  


## References Mentioned in the Presentation

1. K. Saad, S. Kim, B.-I. Lee, and Y. Hwang, *Self-supervised depth estimation and 3D reconstruction with layer-wise LoRA*.
2. Rahima Khanam and Muhammad Hussain, *YOLOv11: An Overview of the Key Architectural Enhancements*, 2024.
3. Ibrahim Majid Mohammed and Nor Ashidi Mat Isa, *Contrast Limited Adaptive Local Histogram Equalization Method for Poor Contrast Image Enhancement*, 2011.
4. Tanakorn Tiay et al., *Flower Recognition System Based on Image Processing*, 2014.
5. Songchenchen Gong and El-Bay Bourennane, *A method based on texture feature and edge detection for people counting in a crowded area*.
6. Pan Zhao and Byeong-Chun Shin, *Detection and Counting of Flowers Based on Digital Images Using Computer Vision and a Concave Point Detection Technique*.
7. Y. Qi, T. Xu, and J. S. Huang, *Analysis of risk management for the coal mine operations*.
8. Silvio Vidal de Miranda Junior et al., *Comparative Study of Depth Anything Model V2 and LiDAR Sensors for Depth Map Estimation in Forest Environment*, 2025.
9. Pan Zhao and Byeong-Chun Shin, *Detection and Counting of Flowers Based on Digital Images Using Computer Vision and a Concave Point Detection Technique*, 2023.
