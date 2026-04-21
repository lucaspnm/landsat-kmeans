# starter_landsat_kmeans_k2_to_k6_save_only.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio
import matplotlib.patches as mpatches

# ----------------------------
# 1) User settings
# ----------------------------
GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
RED_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B4.TIF"
NIR_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B5.TIF"
SWIR1_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B6.TIF"

OUTPUT_DIR = "results"
DOWNSAMPLE = 2
MAX_SAMPLES = 20000
RANDOM_SEED = 0
K_VALUES = [2, 3, 4, 5, 6]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 2) Helper functions
# ----------------------------
def read_band(path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile
    return band, profile

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

def save_false_color(nir, red, green, out_path):
    nir_vis = normalize_for_display(nir)
    red_vis = normalize_for_display(red)
    green_vis = normalize_for_display(green)

    false_color = np.stack([nir_vis, red_vis, green_vis], axis=-1)

    plt.figure(figsize=(12, 10))
    plt.imshow(false_color)
    plt.title("False Color Composite (NIR, Red, Green)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def assign_land_cover_labels(cluster_means):
    """
    More conservative labeling:
    - Avoid overclaiming Urban when desert/bare soil is spectrally similar
    - Use broader, more honest classes
    """
    cluster_to_class = {}

    for i, mean in cluster_means.items():
        green_m, red_m, nir_m, swir1_m, ndvi_m, ndwi_m, ndbi_m = mean
        mean_brightness = (green_m + red_m + nir_m + swir1_m) / 4.0

        # Strong vegetation
        if ndvi_m > 0.5:
            cluster_to_class[i] = "Vegetation"

        # Moderate vegetation
        elif ndvi_m > 0.2:
            cluster_to_class[i] = "Sparse Vegetation"

        # Water-like
        elif ndwi_m > 0.05 and nir_m < 0.15:
            cluster_to_class[i] = "Water"

        # Very dark surfaces
        elif mean_brightness < 0.08:
            cluster_to_class[i] = "Water/Shadow"

        # Low-vegetation surfaces: do NOT force urban
        elif ndvi_m < 0.2 and ndbi_m > 0.0:
            cluster_to_class[i] = "Bare Soil / Urban"

        else:
            cluster_to_class[i] = "Non-Vegetated"

    return cluster_to_class

def build_rgb_map(label_image, cluster_to_class, class_colors):
    rows, cols = label_image.shape
    rgb_map = np.zeros((rows, cols, 3), dtype=np.uint8)

    for cluster_id, class_name in cluster_to_class.items():
        mask = label_image == cluster_id
        rgb_map[mask] = class_colors[class_name]

    rgb_map[label_image == -1] = [0, 0, 0]
    return rgb_map

def save_raw_cluster_map(label_image, k, out_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(label_image, cmap="tab10")
    plt.colorbar(label="Cluster ID")
    plt.title(f"K-means Cluster Map (K = {k})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_labeled_map(rgb_map, cluster_to_class, class_colors, k, out_path):
    used_classes = []
    for cluster_id in sorted(cluster_to_class.keys()):
        cname = cluster_to_class[cluster_id]
        if cname not in used_classes:
            used_classes.append(cname)

    legend_handles = [
        mpatches.Patch(
            color=np.array(class_colors[cname]) / 255.0,
            label=cname
        )
        for cname in used_classes
    ]

    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_map)
    plt.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
    plt.title(f"Land Cover Classification (K = {k})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_feature_scatter(X_train, labels_train, k, out_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_train[:, 1],   # red
        X_train[:, 2],   # nir
        c=labels_train,
        s=4,
        cmap="tab10"
    )
    plt.xlabel("Red Reflectance")
    plt.ylabel("NIR Reflectance")
    plt.title(f"Clusters in Red-NIR Feature Space (K = {k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------
# 3) Load and preprocess data
# ----------------------------
green_dn, profile = read_band(GREEN_PATH)
red_dn, _         = read_band(RED_PATH)
nir_dn, _         = read_band(NIR_PATH)
swir1_dn, _       = read_band(SWIR1_PATH)

if not (green_dn.shape == red_dn.shape == nir_dn.shape == swir1_dn.shape):
    raise ValueError("Input bands do not all have the same shape.")

green = scale_reflectance(green_dn)
red   = scale_reflectance(red_dn)
nir   = scale_reflectance(nir_dn)
swir1 = scale_reflectance(swir1_dn)

green = green[::DOWNSAMPLE, ::DOWNSAMPLE]
red   = red[::DOWNSAMPLE, ::DOWNSAMPLE]
nir   = nir[::DOWNSAMPLE, ::DOWNSAMPLE]
swir1 = swir1[::DOWNSAMPLE, ::DOWNSAMPLE]

print("Downsampled image shape:", green.shape)

save_false_color(
    nir, red, green,
    os.path.join(OUTPUT_DIR, "false_color_composite.png")
)

ndvi = normalized_difference(nir, red)
ndwi = normalized_difference(green, nir)
ndbi = normalized_difference(swir1, nir)

feature_stack = np.stack([green, red, nir, swir1, ndvi, ndwi, ndbi], axis=-1)

rows, cols, num_features = feature_stack.shape
print("Feature stack shape:", feature_stack.shape)

X = feature_stack.reshape(-1, num_features).astype(np.float32)

valid_mask = np.all(np.isfinite(X), axis=1)
raw_valid = np.all((X[:, :4] >= 0.0) & (X[:, :4] <= 1.0), axis=1)
index_valid = np.all((X[:, 4:] >= -1.0) & (X[:, 4:] <= 1.0), axis=1)

valid_mask = valid_mask & raw_valid & index_valid
X_valid = X[valid_mask]

print("Total pixels:", X.shape[0])
print("Valid pixels:", X_valid.shape[0])

rng = np.random.default_rng(RANDOM_SEED)
if X_valid.shape[0] > MAX_SAMPLES:
    idx = rng.choice(X_valid.shape[0], size=MAX_SAMPLES, replace=False)
    X_train = X_valid[idx]
else:
    X_train = X_valid

print("Training samples for k-means:", X_train.shape[0])

# ----------------------------
# 4) Class colors
# ----------------------------
class_colors = {
    "Vegetation":        [34, 139, 34],    # forest green
    "Sparse Vegetation": [154, 205, 50],   # yellow-green
    "Water":             [30, 144, 255],   # dodger blue
    "Water/Shadow":      [65, 105, 225],   # royal blue
    "Bare Soil / Urban": [210, 105, 30],   # chocolate
    "Non-Vegetated":     [210, 180, 140],  # tan
}

# ----------------------------
# 5) Run K = 2 through 6
# ----------------------------
feature_names = ["green", "red", "nir", "swir1", "ndvi", "ndwi", "ndbi"]

for k in K_VALUES:
    print(f"\n=== Running K = {k} ===")

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(X_train)

    labels_train = kmeans.predict(X_train)
    labels_valid = kmeans.predict(X_valid)

    label_image = np.full(X.shape[0], -1, dtype=np.int32)
    label_image[valid_mask] = labels_valid
    label_image = label_image.reshape(rows, cols)

    cluster_means = {}
    for i in range(k):
        cluster_pixels = X_train[labels_train == i]
        if cluster_pixels.size == 0:
            print(f"\nCluster {i}: empty")
            continue

        cluster_mean = cluster_pixels.mean(axis=0)
        cluster_means[i] = cluster_mean

        print(f"\nCluster {i}:")
        for name, val in zip(feature_names, cluster_mean):
            print(f"  {name:>6s}: {val: .4f}")

    cluster_to_class = assign_land_cover_labels(cluster_means)

    print("\nCluster label assignments:")
    for cluster_id, class_name in cluster_to_class.items():
        print(f"  Cluster {cluster_id} -> {class_name}")

    rgb_map = build_rgb_map(label_image, cluster_to_class, class_colors)

    save_raw_cluster_map(
        label_image,
        k,
        os.path.join(OUTPUT_DIR, f"k{k}_raw_cluster_map.png")
    )

    save_labeled_map(
        rgb_map,
        cluster_to_class,
        class_colors,
        k,
        os.path.join(OUTPUT_DIR, f"k{k}_labeled_land_cover_map.png")
    )

    save_feature_scatter(
        X_train,
        labels_train,
        k,
        os.path.join(OUTPUT_DIR, f"k{k}_red_nir_scatter.png")
    )

print(f"\nDone. Saved all outputs to: {OUTPUT_DIR}")