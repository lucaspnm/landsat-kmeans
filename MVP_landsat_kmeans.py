#%%

# starter_landsat_kmeans_local.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio

# ----------------------------
# 1) File paths
# ----------------------------
# Update these to match your downloaded files
GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
RED_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B4.TIF"
NIR_PATH   = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B5.TIF"
SWIR1_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B6.TIF"

# Optional later:
# QA_PIXEL_PATH = "LC08_..._QA_PIXEL.TIF"

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
# 5) Load bands
# ----------------------------
green_dn, profile = read_band(GREEN_PATH)
red_dn, _         = read_band(RED_PATH)
nir_dn, _         = read_band(NIR_PATH)
swir1_dn, _       = read_band(SWIR1_PATH)

# Basic shape check
if not (green_dn.shape == red_dn.shape == nir_dn.shape == swir1_dn.shape):
    raise ValueError("Input bands do not all have the same shape.")

# ----------------------------
# 6) Scale to reflectance
# ----------------------------
green = scale_reflectance(green_dn)
red   = scale_reflectance(red_dn)
nir   = scale_reflectance(nir_dn)
swir1 = scale_reflectance(swir1_dn)

# ----------------------------
# Optional spatial downsampling
# ----------------------------
DOWNSAMPLE = 4   # try 2 or 4

green = green[::DOWNSAMPLE, ::DOWNSAMPLE]
red   = red[::DOWNSAMPLE, ::DOWNSAMPLE]
nir   = nir[::DOWNSAMPLE, ::DOWNSAMPLE]
swir1 = swir1[::DOWNSAMPLE, ::DOWNSAMPLE]

# ----------------------------
# DEBUG: Plot the four bands
# ----------------------------

def show_band(img, title, cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Clip values for better visualization
def clip_for_display(img):
    return np.clip(img, 0, 0.3)

# show_band(clip_for_display(green), "Green (B3)")
# show_band(clip_for_display(red),   "Red (B4)")
# show_band(clip_for_display(nir),   "NIR (B5)")
# show_band(clip_for_display(swir1), "SWIR1 (B6)")


# ----------------------------
# 7) Compute indices
# ----------------------------
ndvi = normalized_difference(nir, red)
ndwi = normalized_difference(green, nir)
ndbi = normalized_difference(swir1, nir)

# ndvi_vis = (ndvi + 1) / 2
# ndwi_vis = (ndwi + 1) / 2
# ndbi_vis = (ndbi + 1) / 2
# 
# fun = np.stack([ndvi_vis, ndwi_vis, ndbi_vis], axis=-1)
# fun = np.clip(fun, 0, 1)
# 
# plt.imshow(fun)
# plt.title("Index-Based Composite (NDVI / NDWI / NDBI)")
# plt.axis("off")
# plt.show()

# ----------------------------
# 8) Build feature stack
# ----------------------------
# Feature vector per pixel:
# [green, red, nir, swir1, ndvi, ndwi, ndbi]
feature_stack = np.stack([green, red, nir, swir1, ndvi, ndwi, ndbi], axis=-1)

rows, cols, num_features = feature_stack.shape
print("Image shape:", (rows, cols))
print("Feature stack shape:", feature_stack.shape)

# Flatten to (num_pixels, num_features)
X = feature_stack.reshape(-1, num_features)

# ----------------------------
# 9) Mask invalid pixels
# ----------------------------
# For MVP:
# - remove NaNs
# - optionally remove extreme reflectance values
valid_mask = np.all(np.isfinite(X), axis=1)

# Optional reflectance sanity check on raw bands only
raw_valid = np.all((X[:, :4] >= 0.0) & (X[:, :4] <= 1.0), axis=1)
index_valid = np.all((X[:, 4:] >= -1.0) & (X[:, 4:] <= 1.0), axis=1)

valid_mask = valid_mask & raw_valid & index_valid

X_valid = X[valid_mask]

print("Total pixels:", X.shape[0])
print("Valid pixels:", X_valid.shape[0])

# ----------------------------
# 10) Optional subsampling for speed
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
# 11) Run k-means
# ----------------------------
K = 5
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
kmeans.fit(X_train)

# Predict labels for all valid pixels in the downsampled image
labels_valid = kmeans.predict(X_valid)

# Rebuild classified image
label_image = np.full(X.shape[0], -1, dtype=np.int32)
label_image[valid_mask] = labels_valid
label_image = label_image.reshape(rows, cols)

# ----------------------------
# 12) Plot classified land cover image
# ----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(label_image, cmap="tab10")
plt.colorbar(label="Cluster")
plt.title("K-means Classified Land Cover Map")
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# 13) Plot feature-space diagnostic
# ----------------------------
labels_train = kmeans.predict(X_train)

plt.figure(figsize=(6, 5))
plt.scatter(
    X_train[:, 1],
    X_train[:, 2],
    c=labels_train,
    s=4,
    cmap="tab10"
)
plt.xlabel("Red reflectance")
plt.ylabel("NIR reflectance")
plt.title("Clusters in Red-NIR Feature Space")
plt.tight_layout()
plt.show()

# ----------------------------
# 14) Print cluster means
# ----------------------------
feature_names = ["green", "red", "nir", "swir1", "ndvi", "ndwi", "ndbi"]

for i in range(K):
    cluster_pixels = X_train[labels_train == i]
    if cluster_pixels.size == 0:
        print(f"\nCluster {i}: empty")
        continue

    cluster_mean = cluster_pixels.mean(axis=0)
    print(f"\nCluster {i}:")
    for name, val in zip(feature_names, cluster_mean):
        print(f"  {name:>6s}: {val: .4f}")