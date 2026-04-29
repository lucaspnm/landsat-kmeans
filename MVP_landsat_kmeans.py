# landsat_kmeans_sweep_simple.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio

GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
RED_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B4.TIF"
NIR_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B5.TIF"
SWIR1_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B6.TIF"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_VALUES = [2, 3, 4, 5, 6]
MAX_SAMPLES = 20000
RANDOM_SEED = 0

# Set to None for full resolution, or use 2/4 for faster testing
DOWNSAMPLE = None

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def scale_reflectance(dn):
    return dn * 0.0000275 - 0.2

def normalized_difference(a, b, eps=1e-6):
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    valid = np.abs(denom) > eps
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out

def normalize_for_display(img, clip_min=0.0, clip_max=0.4):
    img = np.clip(img, clip_min, clip_max)
    return (img - clip_min) / (clip_max - clip_min)

def save_false_color(nir, red, green):
    false_color = np.stack([
        normalize_for_display(nir),
        normalize_for_display(red),
        normalize_for_display(green)
    ], axis=-1)

    plt.figure(figsize=(12, 10))
    plt.imshow(false_color)
    plt.title("False Color Composite (NIR, Red, Green)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "false_color_composite.png"), dpi=300)
    plt.close()

# Load bands
green = scale_reflectance(read_band(GREEN_PATH))
red   = scale_reflectance(read_band(RED_PATH))
nir   = scale_reflectance(read_band(NIR_PATH))
swir1 = scale_reflectance(read_band(SWIR1_PATH))

if DOWNSAMPLE is not None:
    green = green[::DOWNSAMPLE, ::DOWNSAMPLE]
    red   = red[::DOWNSAMPLE, ::DOWNSAMPLE]
    nir   = nir[::DOWNSAMPLE, ::DOWNSAMPLE]
    swir1 = swir1[::DOWNSAMPLE, ::DOWNSAMPLE]

print("Image shape:", green.shape)

save_false_color(nir, red, green)

# Compute indices
ndvi = normalized_difference(nir, red)
ndwi = normalized_difference(green, nir)
ndbi = normalized_difference(swir1, nir)

# Build feature matrix
feature_stack = np.stack([green, red, nir, swir1, ndvi, ndwi, ndbi], axis=-1)
rows, cols, num_features = feature_stack.shape

X = feature_stack.reshape(-1, num_features).astype(np.float32)

valid_mask = np.all(np.isfinite(X), axis=1)
raw_valid = np.all((X[:, :4] >= 0.0) & (X[:, :4] <= 1.0), axis=1)
index_valid = np.all((X[:, 4:] >= -1.0) & (X[:, 4:] <= 1.0), axis=1)

valid_mask = valid_mask & raw_valid & index_valid
X_valid = X[valid_mask]

print("Total pixels:", X.shape[0])
print("Valid pixels:", X_valid.shape[0])

# Sample for training
rng = np.random.default_rng(RANDOM_SEED)

if X_valid.shape[0] > MAX_SAMPLES:
    idx = rng.choice(X_valid.shape[0], size=MAX_SAMPLES, replace=False)
    X_train = X_valid[idx]
else:
    X_train = X_valid

print("Training samples:", X_train.shape[0])

feature_names = ["green", "red", "nir", "swir1", "ndvi", "ndwi", "ndbi"]

for K in K_VALUES:
    print(f"\nRunning K = {K}")

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10)
    labels_train = kmeans.fit_predict(X_train)
    labels_valid = kmeans.predict(X_valid)

    label_image = np.full(X.shape[0], -1, dtype=np.int32)
    label_image[valid_mask] = labels_valid
    label_image = label_image.reshape(rows, cols)

    # Save cluster map with cluster IDs only
    plt.figure(figsize=(12, 10))
    plt.imshow(label_image, cmap="tab10")
    plt.colorbar(label="Cluster ID")
    plt.title(f"K-means Cluster Map, K = {K}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"k{K}_cluster_map.png"), dpi=300)
    plt.close()

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

print(f"\nDone. Figures saved in: {OUTPUT_DIR}")