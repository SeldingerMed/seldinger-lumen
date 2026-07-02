import numpy as np

from lumen.assets.imaging import asset_from_mask, load_npz_volume, segment_threshold
from lumen.sensors import FluoroSensor


def _disk_mask(shape=(36, 36, 16)):
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0:shape[1], 0:shape[0]]
    for z in range(2, 14):
        centers = [(18.0, 18.0)] if z < 8 else [(14.0, 18.0), (22.0, 18.0)]
        radius = 4.0 if z < 8 else 3.0
        for cx, cy in centers:
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            mask[:, :, z] |= disk.T
    return mask


def test_npz_volume_can_be_thresholded_and_imported_as_branching_asset(tmp_path):
    intensity = np.where(_disk_mask(), 240.0, 20.0).astype(np.float32)
    path = tmp_path / "cta.npz"
    np.savez(path, volume=intensity, spacing_mm=np.array([0.5, 0.5, 1.25]),
             origin_mm=np.array([10.0, 20.0, 30.0]))

    vol = load_npz_volume(path)
    mask = segment_threshold(vol, threshold=100.0, foreground="above")
    asset = asset_from_mask(mask.mask, spacing_mm=mask.spacing_mm, origin_mm=mask.origin_mm,
                            min_component_voxels=8)

    assert asset.frame.spacing_mm == (0.5, 0.5, 1.25)
    assert asset.frame.origin_mm == (10.0, 20.0, 30.0)
    assert asset.provenance == "segmented(imported)"
    assert len(asset.edges) >= 3
    assert any(n.id == asset.device_spawn.node_id for n in asset.nodes)
    assert all(np.asarray(edge.R).min() > 0.0 for edge in asset.edges)


def test_fluoro_can_render_tree_wide_asset_roadmap_from_imported_mask():
    asset = asset_from_mask(_disk_mask(), spacing_mm=(0.5, 0.5, 1.0),
                            min_component_voxels=8)
    trunk = next(edge for edge in asset.edges if edge.node_a == asset.device_spawn.node_id)
    wire = np.asarray(trunk.centerline_mm, float)

    scene = FluoroSensor(res=32, nu=48, nv=48, n_samples=80).render_scene(
        wire, radius=0.25, contrast_asset=asset, mu_contrast=0.16)

    assert scene["image"].shape == (48, 48)
    assert scene["masks"]["device"].sum() > 0
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()
