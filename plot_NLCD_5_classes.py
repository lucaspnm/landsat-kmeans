# --------------------------------------------------------------
# plot_nlcd_aligned.py
# --------------------------------------------------------------
# Purpose:
#   - Align the NLCD land‑cover raster to a Landsat‑8 green band.
#   - Collapse the many NLCD categories into a small set of
#     “super‑classes” for easier visual comparison.
#   - Produce two PNGs:
#        1) A colour‑coded map of the collapsed classes (with legend).
#        2) The original NLCD map (uncollapsed) with a dynamic legend.
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.warp import reproject, Resampling

# import helper functions
from landcover_utils import (
    align_nlcd_to_landsat,
    read_band,
    scale_reflectance,
    normalized_difference,
    normalize_for_display,
    save_false_color,
    save_overlay,
    map_nlcd_to_superclass      
)

GREEN_PATH = "LC08_L2SP_041036_20251005_20251115_02_T1_SR_B3.TIF"
NLCD_PATH  = "Annual_NLCD_LndCov_2024_CU_C1V1.tif"

OUTPUT_DIR = "results"

# Ensures the output folder exists (creates it if necessary)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# mapping NLCD numbers to label names - 5 superclasses
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

# Define colormap for the superclasses
CLASS_COLORS = {
    "desert/shrubland": [0, 0, 255],         # blue
    "developed": [255, 0, 0],         # red
    "vegetation": [0, 200, 0],    # green
    "water": [210, 180, 140],    # tan
    "agriculture": [255, 255, 0], # yellow (optional)
    "ignore": [0, 0, 0]          # black
}

# Align and collapse NLCD
green, green_profile = read_band(GREEN_PATH)

nlcd_aligned = align_nlcd_to_landsat(
    NLCD_PATH,
    green_profile,
    dst_shape=green.shape
)

nlcd_super = map_nlcd_to_superclass(nlcd_aligned, NLCD_GROUPS)

print("Aligned NLCD shape:", nlcd_aligned.shape)
print("Unique raw NLCD classes:", np.unique(nlcd_aligned))

# Create RGB map
nlcd_rgb = np.zeros((nlcd_super.shape[0], nlcd_super.shape[1], 3), dtype=np.uint8)

for class_name, color in CLASS_COLORS.items():
    nlcd_rgb[nlcd_super == class_name] = color

# Build legend dynamically
cluster_to_class = {
    0: "desert/shrubland",
    1: "developed",
    2: "vegetation",
    3: "water",
    4: "agriculture",
}

legend_patches = []
used_classes = set(cluster_to_class.values())

for class_name in used_classes:
    color = np.array(CLASS_COLORS[class_name]) / 255.0
    patch = mpatches.Patch(color=color, label=class_name)
    legend_patches.append(patch)

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
cmap = plt.colormaps["tab20"]

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