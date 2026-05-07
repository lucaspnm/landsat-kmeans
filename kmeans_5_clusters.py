import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
import rasterio
from rasterio.warp import reproject, Resampling
from collections import Counter

# import utility functions
from landcover_utils import (
    align_nlcd_to_landsat,
    read_band,
    scale_reflectance,
    normalized_difference,
    normalize_for_display,
    save_false_color,
    save_overlay,
    map_nlcd_to_superclass,
    save_cluster_map
)

# Paths and parameters
GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
RED_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B4.TIF"
NIR_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B5.TIF"
SWIR1_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B6.TIF"
NLCD_PATH  = "Annual_NLCD_LndCov_2024_CU_C1V1.tif"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = 5
MAX_SAMPLES = 20000
RANDOM_SEED = 0

# ------------------------------------------------------------
# Load and preprocess Landsat bands
# ------------------------------------------------------------

# Read the bands and their profiles for georeferencing
green, green_profile = read_band(GREEN_PATH)
red, red_profile = read_band(RED_PATH)
nir, nir_profile = read_band(NIR_PATH)
swir1, swir1_profile = read_band(SWIR1_PATH)

# Convert raw Landsat digital numbers to surface reflectance values
green = scale_reflectance(green)
red = scale_reflectance(red)
nir = scale_reflectance(nir)
swir1 = scale_reflectance(swir1)

# ------------------------------------------------------------
# Align NLCD to the Landsat image grid
# ------------------------------------------------------------

# Reproject the NLCD raster so each NLCD pixel corresponds to a Landsat pixel
nlcd_aligned = align_nlcd_to_landsat(
    NLCD_PATH,
    green_profile,
    dst_shape=green.shape
)

print("Image shape:", green.shape)

# ------------------------------------------------------------
# Save basic scene visualizations
# ------------------------------------------------------------

# Save a false color composite of the Landsat bands for visual reference
save_false_color(nir, red, green)

# Recreate the false color composite for overlaying with NLCD
false_color = np.stack([
    normalize_for_display(nir),
    normalize_for_display(red),
    normalize_for_display(green)
    ], axis=-1)

# Save an overlay of the aligned NLCD on top of the Landsat false color image
# This helps visually verify NLCD/Landsat alignment
save_overlay(false_color, 
             nlcd_aligned, 
             os.path.join(OUTPUT_DIR, "nlcd_overlay.png"))

# ------------------------------------------------------------
# Compute spectral indices
# ------------------------------------------------------------

ndvi = normalized_difference(nir, red)
ndwi = normalized_difference(green, nir)
ndbi = normalized_difference(swir1, nir)

# ------------------------------------------------------------
# Build feature matrix for K-means
# ------------------------------------------------------------

# Stack the raw bands and indices into a 7-layer 3D array (rows, cols, features)
feature_stack = np.stack([green, red, nir, swir1, ndvi, ndwi, ndbi], axis=-1)
rows, cols, num_features = feature_stack.shape

# Reshape from image format (rows, cols, features) to 
# K-means format (pixels, features)
X = feature_stack.reshape(-1, num_features).astype(np.float32)

# Keep only pixels with finite featture values
valid_mask = np.all(np.isfinite(X), axis=1)

# Keep only reasonable reflectance values for the raw bands
raw_valid = np.all((X[:, :4] >= 0.0) & (X[:, :4] <= 1.0), axis=1)

# Keep only valid normalized index values
index_valid = np.all((X[:, 4:] >= -1.0) & (X[:, 4:] <= 1.0), axis=1)

# Final mask combines all validity checks
valid_mask = valid_mask & raw_valid & index_valid
X_valid = X[valid_mask]

print("Total pixels:", X.shape[0])
print("Valid pixels:", X_valid.shape[0])

# ------------------------------------------------------------
# Sample training pixels
# ------------------------------------------------------------

# Use a random subset for fitting K-means to speed up training
rng = np.random.default_rng(RANDOM_SEED)

if X_valid.shape[0] > MAX_SAMPLES:
    idx = rng.choice(X_valid.shape[0], size=MAX_SAMPLES, replace=False)
    X_train = X_valid[idx]
else:
    X_train = X_valid

print("Training samples:", X_train.shape[0])

feature_names = ["green", "red", "nir", "swir1", "ndvi", "ndwi", "ndbi"]

# ------------------------------------------------------------
# Run K-means clustering
# ------------------------------------------------------------

print(f"\nRunning K = {K}")

# Fit K-means on the sampled training pixels
kmeans = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10)
labels_train = kmeans.fit_predict(X_train)

# Predict cluster labels for all valid pixels in the image
labels_valid = kmeans.predict(X_valid)

# ------------------------------------------------------------
# Convert 1D cluster labels back into image format
# ------------------------------------------------------------

# Start with all pixels marked as invalid/background
label_image = np.full(X.shape[0], -1, dtype=np.int32)

# Fill valid pixels with their predicted cluster labels
label_image[valid_mask] = labels_valid

# Reshape back to original image dimensions
label_image = label_image.reshape(rows, cols)

# Save cluster map with legend
cluster_colors = [
    "black",      # -1 background / invalid
    "tab:blue",   # cluster 0
    "tab:red",    # cluster 1
    "tab:green",  # cluster 2
    "tab:cyan",   # cluster 3
    "tab:orange", # cluster 4 if K=5
]

save_cluster_map(
    label_image,
    K,
    cluster_colors,
    os.path.join(OUTPUT_DIR, f"k{K}_cluster_map.png")
)

# Save feature-space scatter
plt.figure(figsize=(8, 6))
plt.scatter(
    X_train[:, 1],
    X_train[:, 2],
    c=labels_train,
    s=4,
    cmap="tab10"
)
plt.xlabel("Red Reflectance")
plt.ylabel("NIR Reflectance")
plt.title(f"Red-NIR Feature Space, K = {K}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"k{K}_red_nir_scatter.png"), dpi=300)
plt.close()

# Print cluster means for interpretation
print(f"Cluster means for K = {K}:")
for cluster_id in range(K):
    cluster_pixels = X_train[labels_train == cluster_id]
    cluster_mean = cluster_pixels.mean(axis=0)

    print(f"\nCluster {cluster_id}:")
    for name, val in zip(feature_names, cluster_mean):
        print(f"  {name:>6s}: {val: .4f}")

# ------------------------------------------------------------
# Collapse NLCD classes into broader evaluation classes
# ------------------------------------------------------------

# Map detailed NLCD numeric class codes to broader land-cover categories
NLCD_GROUPS = {
    11: "water", 
    12: "water",
    
    21: "developed", 
    22: "developed", 
    23: "developed", 
    24: "developed",

    31: "barren",
    
    41: "vegetation",
    42: "vegetation", 
    43: "vegetation",
    90: "vegetation", 
    95: "vegetation", # wetlands mapped to vegetation

    52: "desert/shrubland", 
    71: "desert/shrubland", 

    81: "agriculture", 
    82: "agriculture"
}

# Flatten NLCD and align with valid pixels
nlcd_flat = nlcd_aligned.reshape(-1)
nlcd_valid = nlcd_flat[valid_mask]
nlcd_super = map_nlcd_to_superclass(nlcd_valid, NLCD_GROUPS)

# ------------------------------------------------------------
# Select classes used for quantitative evaluation
# ------------------------------------------------------------

eval_classes = [
    "developed",
    "vegetation",
    "desert/shrubland",
    "agriculture",
    "barren"
]
eval_mask = np.isin(nlcd_super, eval_classes)

labels_eval = labels_valid[eval_mask]
nlcd_eval = nlcd_super[eval_mask]

cluster_to_class = {}

cluster_to_class = {
    0: "desert/shrubland",
    1: "developed",
    2: "vegetation",
    3: "water",
    4: "agriculture",
}

# Convert clusters → predicted classes
predicted_classes = np.array([
    cluster_to_class.get(c, "unknown") for c in labels_valid
])

# Only evaluate valid NLCD classes
valid_eval = np.isin(nlcd_super, eval_classes)

accuracy = np.mean(
    predicted_classes[valid_eval] == nlcd_super[valid_eval]
)

print(f"\nManual mapping accuracy: {accuracy:.4f}")

CLASS_COLORS = {
    "desert/shrubland": [0, 0, 255],         # blue
    "developed": [255, 0, 0],         # red
    "vegetation": [0, 200, 0],    # green
    "water": [210, 180, 140],    # tan
    "agriculture": [255, 255, 0], # yellow (optional)
    "ignore": [0, 0, 0]          # black
}

# Create RGB image
rgb_image = np.zeros((rows, cols, 3), dtype=np.uint8)

# Fill it
for cluster_id, class_name in cluster_to_class.items():
    mask = (label_image == cluster_id)
    rgb_image[mask] = CLASS_COLORS[class_name]

plt.figure(figsize=(12, 10))
plt.imshow(rgb_image)
plt.title("K-means Land Cover Classification (K=5)")
plt.axis("off")

# Build legend dynamically
legend_patches = []
used_classes = set(cluster_to_class.values())

for class_name in used_classes:
    color = np.array(CLASS_COLORS[class_name]) / 255.0
    patch = mpatches.Patch(color=color, label=class_name)
    legend_patches.append(patch)

plt.legend(handles=legend_patches, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "NLCD_k5_labeled_map.png"), dpi=300)
plt.close()