"""Actual routing-descriptor patch attention; never a forged-pixel mask."""

import math

import numpy as np

METHOD = "fadr-route-patch-v1"


def combine_attention(patches, routes):
    """Combine B,L,P patch softmax weights using the same forward's B,L router."""
    patches, routes = np.asarray(patches, dtype=np.float64), np.asarray(routes, dtype=np.float64)
    if patches.ndim != 3 or routes.shape != patches.shape[:2]:
        raise ValueError("Attention shape mismatch")
    side = math.isqrt(patches.shape[2])
    if side != 16 or patches.shape[1] != 4 or side * side != patches.shape[2]:
        raise ValueError("Expected four 16x16 CLIP patch grids")
    if not np.isfinite(patches).all() or not np.isfinite(routes).all() or (patches < 0).any() or (routes < 0).any():
        raise ValueError("Invalid attention weights")
    if not np.allclose(patches.sum(axis=-1), 1, atol=1e-4) or not np.allclose(routes.sum(axis=-1), 1, atol=1e-4):
        raise ValueError("Attention weights must sum to one")
    combined = (patches * routes[:, :, None]).sum(axis=1)
    return [{"method": METHOD, "grid_size": side, "layers": [6, 12, 18, 24],
             "routing_weights": route.tolist(), "patch_weights": grid.reshape(side, side).tolist(),
             "normalization": "per-frame min-max; relative attention, not class attribution"}
            for grid, route in zip(combined, routes)]


def render_attention(payload):
    """Color layer over the exact square face; display normalization only."""
    import cv2

    if payload.get("method") != METHOD:
        raise ValueError("Unknown attention method")
    grid = np.asarray(payload["patch_weights"], dtype=np.float64)
    if grid.shape != (16, 16) or not np.isfinite(grid).all() or (grid < 0).any() or not np.isclose(grid.sum(), 1, atol=1e-4):
        raise ValueError("Invalid patch grid")
    spread = float(np.ptp(grid))
    uniform = spread <= max(1e-10, float(grid.max()) * 1e-6)
    normalized = np.zeros_like(grid) if uniform else (grid - grid.min()) / spread
    # A flat map is transparent, so interpolation cannot invent a hotspot.
    enlarged = cv2.resize(normalized.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.rint(enlarged * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgba = np.dstack([colored, np.full((256, 256), 0 if uniform else 255, dtype=np.uint8)])
    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("Cannot encode attention layer")
    return png.tobytes(), {**payload, "uniform": uniform, "color_map": "TURBO", "display_size": 256}
