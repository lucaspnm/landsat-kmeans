# starter_landsat_kmeans.py

import ee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ----------------------------
# 1) Earth Engine auth/init
# ----------------------------
# Run once interactively if needed:
# ee.Authenticate()

PROJECT_ID = "secure-bongo-392117"   # <-- change this
ee.Initialize(project=PROJECT_ID)

# ----------------------------
# 2) Region of interest
# ----------------------------
# Small box around downtown Los Angeles
roi = ee.Geometry.Rectangle([-118.35, 33.95, -118.10, 34.15])

# ----------------------------
# 3) Load a Landsat 8 L2 scene
# ----------------------------
collection = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(roi)
    .filterDate("2023-01-01", "2023-12-31")
    .sort("CLOUD_COVER")
)

image = ee.Image(collection.first())

# ----------------------------
# 4) Scale surface reflectance bands
#    USGS Landsat C2 L2 SR scale:
#    reflectance = DN * 0.0000275 - 0.2
# ----------------------------
red = image.select("SR_B4").multiply(0.0000275).add(-0.2).rename("red")
nir = image.select("SR_B5").multiply(0.0000275).add(-0.2).rename("nir")
green = image.select("SR_B3").multiply(0.0000275).add(-0.2).rename("green")
swir1 = image.select("SR_B6").multiply(0.0000275).add(-0.2).rename("swir1")

# Optional masking of fill/cloudy pixels can be added later.
# For MVP, keep it simple.

# ----------------------------
# 5) Compute indices
# ----------------------------
ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("ndbi")

# Feature stack
features_img = ee.Image.cat([red, nir, green, swir1, ndvi, ndwi, ndbi]).clip(roi)

# ----------------------------
# 6) Sample pixels into Python
# ----------------------------
# Keep this modest for now so it runs quickly.
sample_fc = features_img.sample(
    region=roi,
    scale=30,
    numPixels=3000,
    geometries=True,
    seed=0
)

print("Sampling from Earth Engine...")
sample_dict = sample_fc.getInfo()
print("Sampling complete.")

# Extract rows
rows = []
coords = []
for feat in sample_dict["features"]:
    props = feat["properties"]
    geom = feat["geometry"]["coordinates"]
    row = [
        props["red"],
        props["nir"],
        props["green"],
        props["swir1"],
        props["ndvi"],
        props["ndwi"],
        props["ndbi"],
    ]
    if all(v is not None for v in row):
        rows.append(row)
        coords.append(geom)

X = np.array(rows, dtype=float)
coords = np.array(coords, dtype=float)

print("Feature matrix shape:", X.shape)

# ----------------------------
# 7) Run k-means in Python
# ----------------------------
K = 5
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)

# ----------------------------
# 8) Simple diagnostic plots
# ----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=5)
plt.xlabel("Red reflectance")
plt.ylabel("NIR reflectance")
plt.title("k-means clusters in feature space")
plt.tight_layout()
plt.show()

# ----------------------------
# 9) Make a rough spatial scatter plot
# ----------------------------
# coords[:, 0] = lon, coords[:, 1] = lat
plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=4)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sampled pixel clusters")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

# ----------------------------
# 10) Optional: inspect cluster means
# ----------------------------
feature_names = ["red", "nir", "green", "swir1", "ndvi", "ndwi", "ndbi"]
for i in range(K):
    cluster_mean = X[labels == i].mean(axis=0)
    print(f"\nCluster {i}:")
    for name, val in zip(feature_names, cluster_mean):
        print(f"  {name:>5s}: {val: .4f}")