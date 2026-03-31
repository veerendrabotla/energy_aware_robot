import numpy as np # type: ignore
from scipy.ndimage import gaussian_filter # type: ignore

def generate_3d_terrain(size=20, terrain_type="mixed"):
    """
    Generates a 3D height map based on terrain type.
    Types: 'flat', 'hills', 'steep', 'mixed'
    """
    if terrain_type == "mixed (flat, hills, steep)":
        terrain_type = "mixed"

    if terrain_type == "flat":
        # Perfectly flat surface — zero elevation everywhere
        terrain = np.zeros((size, size))
    elif terrain_type == "hills":
        # Gentle, smooth rolling hills (heavy Gaussian blur)
        noise = np.random.rand(size, size) * 8.0
        terrain = gaussian_filter(noise, sigma=3.5)
    elif terrain_type == "steep":
        # Sharp, jagged elevation spikes (minimal Gaussian blur)
        noise = np.random.rand(size, size) * 15.0
        terrain = gaussian_filter(noise, sigma=0.8)
    elif terrain_type == "mixed":
        # Combination: gentle base layer + sharp spike overlay
        base_hills = gaussian_filter(np.random.rand(size, size) * 8.0, sigma=3.5)
        sharp_spikes = gaussian_filter(np.random.rand(size, size) * 7.0, sigma=0.8)
        terrain = base_hills + sharp_spikes
    else:
        terrain = np.zeros((size, size))
        
    return terrain

def compute_slope(h1, h2):
    return max(0, h2 - h1)
