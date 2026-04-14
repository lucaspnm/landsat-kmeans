# starter_landsat_kmeans_full_classified.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio
import matplotlib.patches as mpatches

# ----------------------------
# 1) File paths
# ----------------------------
GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
RED_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B4.TIF"
NIR_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B5.TIF"
SWIR1_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B6.TIF"

# ----------------------------
# 2) Read a single-band GeoTIFF
# ----------------------------
def read_band(path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile
    return band, profile

# ----------------------------
# 3) Landsat C2 L2 scaling
#    reflectance = DN * 0.0000275 - 0.2
# ----------------------------
def scale_reflectance(dn):
    return dn * 0.0000275 - 0.2

# ----------------------------
# 4) Safe normalized difference
# ----------------------------
def normalized_difference(a, b, eps=1e-6):
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    valid = np.abs(denom) > eps
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out

# ----------------------------
# 5) Display helpers
# ----------------------------
def normalize_for_display(img, clip_min=0.0, clip_max=0.4):
    img = np.clip(img, clip_min, clip_max)
    return (img - clip_min) / (clip_max - clip_min)

def show_false_color(nir, red, green):
    nir_vis = normalize_for_display(nir)
    red_vis = normalize_for_display(red)
    green_vis = normalize_for_display(green)

    false_color = np.stack([nir_vis, red_vis, green_vis], axis=-1)

    plt.figure(figsize=(8, 8))
    plt.imshow(false_color)
    plt.title("False Color Composite (NIR, Red, Green)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 6) Load bands
# ----------------------------
green_dn, profile = read_band(GREEN_PATH)
red_dn, _         = read_band(RED_PATH)
nir_dn, _         = read_band(NIR_PATH)
swir1_dn, _       = read_band(SWIR1_PATH)

if not (green_dn.shape == red_dn.shape == nir_dn.shape == swir1_dn.shape):
    raise ValueError("Input bands do not all have the same shape.")

# ----------------------------
# 7) Scale to reflectance
# ----------------------------
green = scale_reflectance(green_dn)
red   = scale_reflectance(red_dn)
nir   = scale_reflectance(nir_dn)
swir1 = scale_reflectance(swir1_dn)

# ----------------------------
# 8) Optional downsampling
# ----------------------------
# Recommended: 2 or 4
DOWNSAMPLE = 2

green = green[::DOWNSAMPLE, ::DOWNSAMPLE]
red   = red[::DOWNSAMPLE, ::DOWNSAMPLE]
nir   = nir[::DOWNSAMPLE, ::DOWNSAMPLE]
swir1 = swir1[::DOWNSAMPLE, ::DOWNSAMPLE]

print("Downsampled image shape:", green.shape)

# ----------------------------
# 9) False color image
# ----------------------------
show_false_color(nir, red, green)

# ----------------------------
# 10) Compute indices
# ----------------------------
ndvi = normalized_difference(nir, red)
ndwi = normalized_difference(green, nir)
ndbi = normalized_difference(swir1, nir)

# ----------------------------
# 11) Build feature stack
# ----------------------------
# Feature order:
# [green, red, nir, swir1, ndvi, ndwi, ndbi]
feature_stack = np.stack([green, red, nir, swir1, ndvi, ndwi, ndbi], axis=-1)

rows, cols, num_features = feature_stack.shape
print("Feature stack shape:", feature_stack.shape)

X = feature_stack.reshape(-1, num_features)

# ----------------------------
# 12) Mask invalid pixels
# ----------------------------
valid_mask = np.all(np.isfinite(X), axis=1)

raw_valid = np.all((X[:, :4] >= 0.0) & (X[:, :4] <= 1.0), axis=1)
index_valid = np.all((X[:, 4:] >= -1.0) & (X[:, 4:] <= 1.0), axis=1)

valid_mask = valid_mask & raw_valid & index_valid
X_valid = X[valid_mask]

print("Total pixels:", X.shape[0])
print("Valid pixels:", X_valid.shape[0])

# ----------------------------
# 13) Sample training pixels
# ----------------------------
MAX_SAMPLES = 20000
rng = np.random.default_rng(0)

if X_valid.shape[0] > MAX_SAMPLES:
    idx = rng.choice(X_valid.shape[0], size=MAX_SAMPLES, replace=False)
    X_train = X_valid[idx]
else:
    X_train = X_valid

print("Training samples for k-means:", X_train.shape[0])

# ----------------------------
# 14) Run k-means
# ----------------------------
K = 5
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
kmeans.fit(X_train)

# Labels for training sample
labels_train = kmeans.predict(X_train)

# Labels for all valid pixels in downsampled image
labels_valid = kmeans.predict(X_valid)

# ----------------------------
# 15) Rebuild cluster image
# ----------------------------
label_image = np.full(X.shape[0], -1, dtype=np.int32)
label_image[valid_mask] = labels_valid
label_image = label_image.reshape(rows, cols)

# ----------------------------
# 16) Inspect cluster means
# ----------------------------
feature_names = ["green", "red", "nir", "swir1", "ndvi", "ndwi", "ndbi"]

cluster_means = {}

for i in range(K):
    cluster_pixels = X_train[labels_train == i]
    if cluster_pixels.size == 0:
        print(f"\nCluster {i}: empty")
        continue

    cluster_mean = cluster_pixels.mean(axis=0)
    cluster_means[i] = cluster_mean

    print(f"\nCluster {i}:")
    for name, val in zip(feature_names, cluster_mean):
        print(f"  {name:>6s}: {val: .4f}")

# ----------------------------
# 17) Assign land cover names
# ----------------------------
# Simple rule-based interpretation from cluster means
cluster_to_class = {}

for i, mean in cluster_means.items():
    green_m, red_m, nir_m, swir1_m, ndvi_m, ndwi_m, ndbi_m = mean

    # Prioritize strong vegetation first
    if ndvi_m > 0.5:
        cluster_to_class[i] = "Vegetation"

    # Water if NDWI is clearly positive and reflectance is dark
    elif ndwi_m > 0.05 and nir_m < 0.15:
        cluster_to_class[i] = "Water"

    # Built-up if NDBI is elevated and NDVI is low
    elif ndbi_m > 0.08 and ndvi_m < 0.25:
        cluster_to_class[i] = "Urban"

    # Moderate vegetation
    elif ndvi_m > 0.2:
        cluster_to_class[i] = "Sparse Vegetation"

    # Very dark cluster
    elif (green_m + red_m + nir_m + swir1_m) / 4 < 0.08:
        cluster_to_class[i] = "Water/Shadow"

    else:
        cluster_to_class[i] = "Bare Soil"

print("\nCluster label assignments:")
for cluster_id, class_name in cluster_to_class.items():
    print(f"  Cluster {cluster_id} -> {class_name}")

# ----------------------------
# 18) Define colors for land cover classes
# ----------------------------
class_colors = {
    "Vegetation":        [34, 139, 34],    # forest green
    "Sparse Vegetation": [154, 205, 50],   # yellow-green
    "Urban":             [220, 20, 60],    # crimson
    "Bare Soil":         [210, 180, 140],  # tan
    "Water":             [30, 144, 255],   # dodger blue
    "Water/Shadow":      [65, 105, 225],   # royal blue
}

# ----------------------------
# 19) Convert cluster image to RGB land cover map
# ----------------------------
rgb_map = np.zeros((rows, cols, 3), dtype=np.uint8)

for cluster_id, class_name in cluster_to_class.items():
    mask = label_image == cluster_id
    rgb_map[mask] = class_colors[class_name]

# Invalid pixels stay black
rgb_map[label_image == -1] = [0, 0, 0]

# ----------------------------
# 20) Plot raw cluster image
# ----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(label_image, cmap="tab10")
plt.colorbar(label="Cluster ID")
plt.title("K-means Cluster Map")
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# 21) Plot labeled land cover map
# ----------------------------
plt.figure(figsize=(10, 8))
plt.imshow(rgb_map)
plt.title("Land Cover Classification (K-means)")
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# 22) Plot labeled land cover map with legend
# ----------------------------
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

plt.figure(figsize=(10, 8))
plt.imshow(rgb_map)
plt.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
plt.title("Land Cover Classification (K-means)")
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# 23) Feature-space scatter
# ----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(
    X_train[:, 1],   # red
    X_train[:, 2],   # nir
    c=labels_train,
    s=4,
    cmap="tab10"
)
plt.xlabel("Red Reflectance")
plt.ylabel("NIR Reflectance")
plt.title("Clusters in Red-NIR Feature Space")
plt.tight_layout()
plt.show()