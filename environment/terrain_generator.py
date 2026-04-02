import numpy as np # type: ignore
from scipy.ndimage import gaussian_filter # type: ignore

def generate_3d_terrain(size=20, terrain_type="mixed"):
    """
    Generates a 3D height map based on terrain type.
    Types: 'flat', 'hills', 'steep', 'mixed'
    
    All terrain is normalized so that the maximum adjacent-cell height
    difference (slope) is controlled relative to MAX_SLOPE (0.5):
      - flat:  zero elevation
      - hills: max slope ≈ 0.25 (well under threshold, fully navigable)
      - steep: max slope ≈ 0.45 (close to threshold, challenging but navigable)
      - mixed: max slope ≈ 0.40 (moderate challenge)
    """
    if terrain_type == "mixed (flat, hills, steep)":
        terrain_type = "mixed"

    if terrain_type == "flat":
        # Perfectly flat surface — zero elevation everywhere
        terrain = np.zeros((size, size))
    elif terrain_type == "hills":
        # Gentle, smooth rolling hills
        noise = np.random.rand(size, size)
        terrain = gaussian_filter(noise, sigma=3.0)
        terrain = _normalize_slopes(terrain, target_max_slope=0.25)
    elif terrain_type == "steep":
        # Sharp elevation changes with tighter smoothing
        noise = np.random.rand(size, size)
        terrain = gaussian_filter(noise, sigma=1.2)
        terrain = _normalize_slopes(terrain, target_max_slope=0.45)
    elif terrain_type == "mixed":
        # Gentle base + sharper overlay
        base_hills = gaussian_filter(np.random.rand(size, size), sigma=3.0)
        sharp_spikes = gaussian_filter(np.random.rand(size, size), sigma=1.2)
        terrain = base_hills + sharp_spikes
        terrain = _normalize_slopes(terrain, target_max_slope=0.35)
    else:
        terrain = np.zeros((size, size))
        
    return terrain


def _normalize_slopes(terrain, target_max_slope=0.40):
    """
    Scales the terrain so that the maximum adjacent-cell height difference
    (in any direction) equals target_max_slope. This guarantees navigability
    while preserving the shape/character of the terrain.
    """
    # Calculate all adjacent-cell differences (up, down, left, right)
    dy = np.abs(np.diff(terrain, axis=0))  # vertical neighbours
    dx = np.abs(np.diff(terrain, axis=1))  # horizontal neighbours
    
    current_max = max(np.max(dy) if dy.size > 0 else 0,
                      np.max(dx) if dx.size > 0 else 0)
    
    if current_max > 0:
        scale_factor = target_max_slope / current_max
        # Center around the mean, scale, then shift back to positive
        mean_val = np.mean(terrain)
        terrain = (terrain - mean_val) * scale_factor + mean_val
        terrain = terrain - np.min(terrain)  # ensure no negative heights
    
    return terrain


def compute_slope(h1, h2):
    return max(0, h2 - h1)
