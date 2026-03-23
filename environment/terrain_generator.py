import numpy as np
from scipy.ndimage import gaussian_filter

def generate_3d_terrain(size=20, terrain_type="mixed"):
    """
    Generates a 3D height map based on terrain type.
    Types: 'flat', 'hills', 'steep', 'mixed'
    """
    if terrain_type == "flat":
        terrain = np.random.rand(size, size) * 0.1
    elif terrain_type == "hills":
        noise = np.random.rand(size, size) * 5.0
        terrain = gaussian_filter(noise, sigma=2.0)
    elif terrain_type == "steep":
        noise = np.random.rand(size, size) * 15.0
        terrain = gaussian_filter(noise, sigma=1.0)
    elif terrain_type == "mixed":
        noise1 = np.random.rand(size, size) * 10.0
        noise2 = np.random.rand(size, size) * 5.0
        terrain = gaussian_filter(noise1, sigma=3.0) + gaussian_filter(noise2, sigma=1.0)
    else:
        terrain = np.zeros((size, size))
        
    return terrain

def compute_slope(h1, h2):
    return max(0, h2 - h1)
