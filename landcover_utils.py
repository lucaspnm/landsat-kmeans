import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
import rasterio
from rasterio.warp import reproject, Resampling
from collections import Counter

OUTPUT_DIR = "results"

def align_nlcd_to_landsat(nlcd_path, landsat_profile, dst_shape):
    """
    Reproject the NLCD Land-cover image to match the Landsat image grid.
    Returns aligned NLCD as a NumPy array.
    """
    
    # Allocate an an empty destination array with the same shape as 
    # the Landsat image and a suitable data type for the NLCD class values
    nlcd_aligned = np.zeros(dst_shape, dtype=np.int16)

    # Open the original NLCD raster and reproject it onto the Landsat grid
    with rasterio.open(nlcd_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=nlcd_aligned,

            # Source georeferencing information from the NLCD file
            src_transform=src.transform,
            src_crs=src.crs,

            # Destination georeferencing information from the Landsat profile
            dst_transform=landsat_profile["transform"],
            dst_crs=landsat_profile["crs"],
            
            # For each output pixel label, use the value of the nearest 
            # input pixel (i.e. no interpolation)
            resampling=Resampling.nearest
        )
    return nlcd_aligned

def read_band(path):
    """
    Read in the first band of the raster as a 2D array and 
    convert to float32 for processing. Also return the profile for later 
    use in georeferencing.
    """

    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile
        return band, profile

def scale_reflectance(dn):
    """
    Convert Landsat surface reflectance digital numbers to reflectance values.
    """

    # Landsat Collection 2 Level 2 scale factor and offset.
    return dn * 0.0000275 - 0.2

def normalized_difference(a, b, eps=1e-6):
    """
    Compute a normalized difference index between two bands.
    """

    # Compute the denominator seperately so invalid pixels 
    # can be masked out (e.g. where both a and b are zero, or where either is NaN)
    denom = a + b
    
    # Initialize output with NaNs, then fill in valid pixels
    out = np.full_like(a, np.nan, dtype=np.float32)

    # Avoid division by zero and invalid values by only computing 
    # the index where the denominator is sufficiently large
    valid = np.abs(denom) > eps
    
    # Compute the normalized difference for valid pixels only
    out[valid] = (a[valid] - b[valid]) / denom[valid]

    return out

def normalize_for_display(img, clip_min=0.0, clip_max=0.4):
    """
    Clip and normalize an image for display purposes. This is useful for 
    visualizing reflectance bands which may have a small range of values.  
    The default clip range is chosen to highlight typical surface reflectance values,
    but can be adjusted as needed.
    """

    img = np.clip(img, clip_min, clip_max)
    return (img - clip_min) / (clip_max - clip_min)

def save_false_color(nir, red, green):
    """
    Create and save a false color composite image using the NIR, Red, and Green bands.
    The NIR band is mapped to red, the Red band to green, and the Green band to blue 
    for visualization. The output is saved as a PNG file.
    """

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

def save_overlay(false_color, nlcd_aligned, out_path):
    """
    Save an overlay of the aligned NLCD image on top of the Landsat 
    false color composite image.
    """

    # Plot the Landsat false color image as the base layer
    plt.figure(figsize=(12, 10))
    plt.imshow(false_color)

    # Overlay the aligned NLCD image with partial transparency
    plt.imshow(nlcd_aligned, cmap="tab20", alpha=0.35)

    # Save the figure with a title and no axes for a clean look
    plt.title("NLCD Overlay on Landsat")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def map_nlcd_to_superclass(nlcd_array, nlcd_groups):
    """
    Collapse the original NLCD classes into broader superclasses based on the 
    provided mapping from nlcd_groups.   
    """

    # Start all pixels as ignored in case an NLCD code is not included in the mapping
    mapped = np.full(nlcd_array.shape, "ignore", dtype=object)

    # Replace each NLCD numeric code with its corresponding superclass label
    for key, val in nlcd_groups.items():
        mapped[nlcd_array == key] = val

    return mapped

def save_cluster_map(label_image, K, cluster_colors, output_path):
    """
    Create a colormap for background + clusters, plot the cluster map, 
    and save it with a legend.
    """

    # Create a colormap for background + clusters
    cmap = ListedColormap(cluster_colors[:K + 1])
    bounds = np.arange(-1.5, K + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 10))
    img = plt.imshow(label_image, cmap=cmap, norm=norm)
    plt.title(f"K-means Cluster Map, K = {K}")
    plt.axis("off")

    # Build legend 
    legend_patches = [
        mpatches.Patch(color="black", label="Background / invalid")
    ]

    for cluster_id in range(K):
        legend_patches.append(
            mpatches.Patch(
                color=cluster_colors[cluster_id + 1], 
                label=f"Cluster {cluster_id}"
            )
        )

    plt.legend(handles=legend_patches, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"k{K}_cluster_map.png"), dpi=300)
    plt.close()
