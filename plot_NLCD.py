import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Path to the NLCD GeoTIFF
NLCD_PATH = "Annual_NLCD_LndCov_2024_CU_C1V1.tif"

# Desired output size – e.g. 2000 × 3000 pixels (adjust as needed)
out_width, out_height = 3000, 2000

nlcd_colors = {
    11: "#4674b4",  # Open Water
    21: "#d1def8",  # Developed, Open Space
    22: "#b6c8e9",  # Developed, Low Intensity
    23: "#84a6d6",  # Developed, Medium Intensity
    24: "#5275b4",  # Developed, High Intensity
    31: "#d1e5a8",  # Barren Land (Rock/Sand/Clay)
    41: "#c8e6c9",  # Deciduous Forest
    42: "#91c287",  # Evergreen Forest
    43: "#679b63",  # Mixed Forest
    52: "#cfd58e",  # Shrub/Scrub
    71: "#f2d889",  # Grassland/Herbaceous
    81: "#dfc27d",  # Pasture/Hay
    82: "#c99572",  # Cultivated Crops
    90: "#d4c9c9",  # Woody Wetlands
    95: "#a68c8c",  # Emergent Herbaceous Wetlands
    96: "#c0c0c0",  # Snow/Ice (rare in NLCD)
}

# Build a ListedColormap ordered by class value
sorted_vals = sorted(nlcd_colors)
cmap = mcolors.ListedColormap([nlcd_colors[v] for v in sorted_vals])

# Normalisation so each integer class gets its own colour band
bounds = sorted_vals + [sorted_vals[-1] + 1]          # add an upper edge
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Read the raster at the reduced resolution
with rasterio.open(NLCD_PATH) as src:
    # Compute the scaling factors
    scale_x = src.width / out_width
    scale_y = src.height / out_height

    # Read the band while resampling to the smaller shape
    data = src.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.nearest   # keep the original class values
    )

    # Update the transform so the plot is georeferenced correctly
    transform = src.transform * src.transform.scale(
        (src.width / out_width),
        (src.height / out_height)
    )


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
img = ax.imshow(
    data,
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    extent=(
        transform[2],
        transform[2] + transform[0] * out_width,
        transform[5] + transform[4] * out_height,
        transform[5],
    ),
)

ax.set_title("NLCD 2024 Land‑Cover (down‑sampled)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Legend
handles = [mpatches.Patch(color=nlcd_colors[v], label=str(v)) for v in sorted_vals]
ax.legend(
    handles=handles,
    title="NLCD class",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
    frameon=False,
)

plt.tight_layout()
plt.show()