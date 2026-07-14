"""Asset schema round-trip + lumen-field behaviour."""

import numpy as np

from lumen.assets import procedural
from lumen.assets.schema import Asset
from lumen.core.lumen_field import LumenField


def test_make_demo_assets_writes_to_requested_directory(tmp_path):
    from examples import make_demo_assets

    out = tmp_path / "demo_assets"
    make_demo_assets.main(out)

    assert sorted(p.name for p in out.iterdir()) == [
        "bifurcation.json",
        "stenotic_tube.json",
        "straight_tube.json",
        "tortuous_tree.json",
        "tortuous_tube.json",
    ]
    for path in out.iterdir():
        assert Asset.load(str(path)).provenance == "procedural"


def test_asset_roundtrip(tmp_path):
    asset = procedural.bifurcation()
    path = tmp_path / "case.json"
    asset.save(str(path))
    back = Asset.load(str(path))
    assert back.provenance == "procedural"
    assert len(back.edges) == 3
    # geometry survives the round trip
    e0a = np.asarray(asset.edges[0].centerline_mm)
    e0b = np.asarray(back.edges[0].centerline_mm)
    assert np.allclose(e0a, e0b)


def test_stenosis_narrows():
    lf = LumenField.stenosis(length=100, radius=2.0, at=50, severity=0.6)
    assert lf.eval(0.0) > lf.eval(50.0)        # narrower at the stenosis
    assert lf.eval(50.0) < 2.0 * (1 - 0.5)     # at least ~half occluded at the dip


def test_emitted_asset_is_procedural():
    # the firewall depends on this invariant
    for a in (procedural.straight_tube(), procedural.stenotic_tube(),
              procedural.tortuous_tube(), procedural.bifurcation(),
              procedural.tortuous_tree()):
        assert a.provenance == "procedural"


def test_lumenfield_rejects_partial_theta_grid():
    # #15 — eval() wraps theta periodically; a partial theta grid (not a full 2π
    # revolution) would be silently wrong, so it must be rejected.
    import pytest
    with pytest.raises(ValueError):
        LumenField(np.array([0.0, 1.0]), np.array([0.0, np.pi / 2]), np.ones((2, 2)))
    # a full revolution (endpoint-excluded) is accepted
    th = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    LumenField(np.array([0.0, 1.0]), th, np.ones((2, 8)))
