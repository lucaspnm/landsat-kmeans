# plot_nlcd_aligned.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.warp import reproject, Resampling

# Paths
GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
NLCD_PATH  = "Annual_NLCD_LndCov_2024_CU_C1V1.tif"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# mapping NLCD numbers to label names
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

CLASS_COLORS = {
    "desert/shrubland": [0, 0, 255],         # blue
    "developed": [255, 0, 0],         # red
    "vegetation": [0, 200, 0],    # green
    "water": [210, 180, 140],    # tan
    "agriculture": [255, 255, 0], # yellow (optional)
    "ignore": [0, 0, 0]          # black
}

# Helpers
def align_nlcd_to_landsat(nlcd_path, landsat_path):
    with rasterio.open(landsat_path) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs

    nlcd_aligned = np.zeros(dst_shape, dtype=np.int16)

    with rasterio.open(nlcd_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=nlcd_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    return nlcd_aligned

def map_nlcd_to_superclass(nlcd_array):
    mapped = np.full(nlcd_array.shape, "ignore", dtype=object)

    for code, class_name in NLCD_GROUPS.items():
        mapped[nlcd_array == code] = class_name

    return mapped

# Align and collapse NLCD
nlcd_aligned = align_nlcd_to_landsat(NLCD_PATH, GREEN_PATH)
nlcd_super = map_nlcd_to_superclass(nlcd_aligned)

print("Aligned NLCD shape:", nlcd_aligned.shape)
print("Unique raw NLCD classes:", np.unique(nlcd_aligned))

# Create RGB map
nlcd_rgb = np.zeros((nlcd_super.shape[0], nlcd_super.shape[1], 3), dtype=np.uint8)

for class_name, color in CLASS_COLORS.items():
    nlcd_rgb[nlcd_super == class_name] = color

# Plot with discrete legend
legend_classes = [
    "developed",
    "vegetation",
    "desert/shrubland",
    "agriculture",
    "barren"
]

legend_patches = [
    mpatches.Patch(
        color=np.array(CLASS_COLORS[class_name]) / 255.0,
        label=class_name
    )
    for class_name in legend_classes
]

plt.figure(figsize=(12, 10))
plt.imshow(nlcd_rgb)
plt.title("Collapsed NLCD Reference Map")
plt.axis("off")
plt.legend(handles=legend_patches, loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "nlcd_collapsed_reference.png"), dpi=300)
plt.close()

print("Saved:", os.path.join(OUTPUT_DIR, "nlcd_collapsed_reference.png"))

# ----------------------------
# Plot raw NLCD with legend
# ----------------------------

# NLCD class names (subset of standard legend)
NLCD_CLASS_NAMES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}

# Mask background (optional)
nlcd_plot = np.where(nlcd_aligned == 0, np.nan, nlcd_aligned)

plt.figure(figsize=(12, 10))
im = plt.imshow(nlcd_plot, cmap="tab20")
plt.title("Raw NLCD Land Cover (Uncollapsed)")
plt.axis("off")

# Find which classes actually appear
unique_classes = np.unique(nlcd_aligned)
unique_classes = unique_classes[unique_classes > 0]  # remove background

# Build legend dynamically
legend_patches = []
cmap = plt.cm.get_cmap("tab20")

for i, cls in enumerate(unique_classes):
    color = cmap(i % 20)  # cycle through tab20 colors
    label = NLCD_CLASS_NAMES.get(cls, f"Class {cls}")
    legend_patches.append(
        mpatches.Patch(color=color, label=label)
    )

plt.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=8
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "nlcd_raw_with_legend.png"), dpi=300)
plt.close()

print("Saved raw NLCD plot with legend")