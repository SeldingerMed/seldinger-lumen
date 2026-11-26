"""The viewer, the shipped cases and compact meshes."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from lumen import cases
from lumen.assets import mesh
from lumen.viewer import page, record
from lumen.viewer.scene import resolve, scene_dict, tube_mesh


def _signed_volume(v, f):
    p = np.asarray(v, float)[np.asarray(f)]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


# ------------------------------------------------------------------ shipped cases
def test_every_shipped_case_loads_and_credits_an_open_licence():
    ids = cases.names()
    assert len(ids) >= 4
    for cid in ids:
        card = cases.card(cid)
        assert card["licence"] in ("CC0", "CC BY 4.0", "CC BY-SA 4.0", "CC BY 3.0")
        assert card["citation"] and card["source_url"].startswith("https://")
        asset, names = cases.load(cid)
        assert len(asset.edges) >= 3
        assert card["vessels"], cid
        for vessel in card["vessels"]:                       # every listed target is reachable
            assert names["vessels"][vessel]["target_node"] in {n.id for n in asset.nodes}
        v, f = mesh.load_npz(cases.case_folder(cid) / "surface.npz")
        assert len(f) > 1000 and _signed_volume(v, f) > 0       # closed, facing out


def test_shipped_cases_are_small():
    for cid in cases.names():
        size = sum(p.stat().st_size for p in cases.case_folder(cid).iterdir())
        assert size < 3_000_000, (cid, size)


def test_case_names_cannot_escape_the_cases_folder():
    for bad in ("../lumen", "..", "/etc", "", ".hidden", "a/b"):
        assert cases.case_folder(bad) is None
    with pytest.raises(KeyError):
        cases.card("no-such-case")


def _solver():
    pytest.importorskip("warp")
    pytest.importorskip("newton")


def test_a_shipped_case_drives_like_an_import():
    _solver()
    env = cases.make_env("head-tof-mra", "right middle cerebral artery", max_steps=3)
    _, _, _, _, info = env.step(np.array([1.0, 0.0], np.float32))
    assert np.isfinite(info["dist"])


# ------------------------------------------------------------------ meshes
def test_compact_surface_round_trips_within_storage_precision(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.uniform(-80, 120, (500, 3)).astype(np.float32)
    f = rng.integers(0, 500, (900, 3)).astype(np.uint32)
    mesh.save_npz(tmp_path / "s.npz", v, f)
    v2, f2 = mesh.load_npz(tmp_path / "s.npz")
    assert np.array_equal(f, f2)
    assert np.abs(v - v2).max() <= mesh.QUANTUM_MM / 2 + 1e-4


def test_simplify_keeps_shape_and_drops_triangles():
    a = resolve("arch").asset
    v, f = tube_mesh(a, n_theta=32, step_mm=0.5)
    v2, f2 = mesh.simplify_to(v, f, len(f) // 3)
    assert len(f2) <= len(f) // 3
    lo, hi = v.min(0), v.max(0)
    assert np.allclose(v2.min(0), lo, atol=2.0) and np.allclose(v2.max(0), hi, atol=2.0)
    assert _signed_volume(v2, f2) == pytest.approx(_signed_volume(v, f), rel=0.1)


def test_generated_tubes_are_closed_and_face_outward():
    v, f = tube_mesh(resolve("arch").asset)
    assert _signed_volume(v, f) > 0
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()                                  # watertight: no open ends


# ------------------------------------------------------------------ scenes and pages
def test_generated_anatomy_gets_names_to_aim_at():
    a = resolve("arch")
    assert {"carotid", "subclavian", "brachio", "descending"} <= set(a.names["vessels"])


def test_unknown_anatomy_lists_what_is_available():
    with pytest.raises(ValueError, match="head-tof-mra"):
        resolve("no-such-anatomy")


def test_page_is_self_contained_and_carries_the_scene():
    html = page.build("head-tof-mra")
    assert "<script src" not in html and "http://" not in html.split("lumen-data")[0]
    data = json.loads(re.search(r'id="lumen-data">(.*?)</script>', html, re.S).group(1))
    s = data["scene"]
    assert s["mesh"]["kind"] == "scan surface" and s["mesh"]["triangle_count"] > 1000
    assert any(lbl["name"] == "right middle cerebral artery" for lbl in s["labels"])
    assert s["source"]["licence"] == "CC0"


def test_every_code_folder_is_a_regular_package_so_wheels_include_it():
    # setuptools' find skips folders without an __init__.py, so their modules are left out of
    # the wheel; on Python 3.10/3.11 such a folder's resources are also a MultiplexedPath.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "lumen"
    missing = sorted(str(d.relative_to(root)) for d in {p.parent for p in root.rglob("*.py")}
                     if not (d / "__init__.py").exists())
    assert missing == []


def test_names_cannot_break_out_of_the_data_block():
    a = resolve("arch")
    a.label = "x</script><script>alert(1)</script>"
    html = page.build(a)
    assert html.count("</script>") == 2                          # the data block and the code


def test_scene_mesh_is_capped_for_the_page():
    from lumen.viewer import scene as sc

    s = scene_dict(resolve("head-neck-cta"))
    assert s["mesh"]["triangle_count"] <= sc.MAX_PAGE_TRIANGLES


# ------------------------------------------------------------------ runs
def test_a_recorded_run_replays_and_reloads(tmp_path):
    _solver()
    from lumen.envs.tree_nav import TreeNavEnv

    a = resolve("arch")
    env = TreeNavEnv(a.asset, target_node="carotid_out", max_steps=12)
    run = record.record(env, steps=12, anatomy="arch", target="carotid")
    assert run["frames"][0]["step"] == 0 and len(run["frames"]) == 13
    assert np.asarray(run["frames"][-1]["nodes"]).shape[1] == 3
    p = record.save(run, tmp_path / "run.json")
    again = record.load(p)
    assert again["outcome"] == run["outcome"]
    html = page.build(a, again)
    assert '"replay":{' in html


def test_a_damaged_replay_is_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": record.REPLAY_VERSION,
                             "frames": [{"step": 0, "nodes": [[0, 0, "x"]]}]}))
    with pytest.raises(ValueError):
        record.load(p)
    p.write_text("{}")
    with pytest.raises(ValueError, match="not a Lumen replay"):
        record.load(p)


def test_follow_policy_backs_off_and_turns_when_stuck():
    pol = record.FollowPolicy(stall_steps=2, max_back=2, turn_steps=1)
    acts = [tuple(pol({"dist": 10.0, "off_route": False})) for _ in range(7)]
    assert (1.0, 0.0) in acts and (-1.0, 0.0) in acts and (0.0, 1.0) in acts


def test_view_command_writes_a_page(tmp_path, capsys):
    from lumen.cli import view_main

    out = tmp_path / "v.html"
    view_main(["arch", "--out", str(out)])
    assert out.exists() and out.stat().st_size > 10_000
    assert "Viewer written" in capsys.readouterr().out


def test_cases_command_lists_credits(capsys):
    from lumen.cli import cases_main

    cases_main([])
    text = capsys.readouterr().out
    assert "head-tof-mra" in text and "CC0" in text and "CC BY 4.0" in text


@pytest.mark.parametrize("look", ["monitor", "anatomy", "glass"])
def test_each_look_is_offered_and_chosen_at_start(look):
    from lumen.viewer.page import build

    html = build("arch", look=look)
    assert f'"look":"{look}"' in html
    for name in ("monitor", "anatomy", "glass"):
        assert f'<option value="{name}">' in html and f"{name}: {{" in html


def test_an_unknown_look_is_refused():
    from lumen.viewer.page import build

    with pytest.raises(ValueError):
        build("arch", look="neon")


# ------------------------------------------------------------------ review round 2
@pytest.mark.parametrize("argv, message", [
    (["arch", "--replay", "r.json", "--run"], "cannot be combined"),
    (["arch", "--save-run", "r.json"], "--save-run needs --run"),
    (["arch", "--run", "--target", "x", "--steps", "0"], "positive whole number"),
])
def test_view_refuses_contradictory_options(argv, message, capsys):
    from lumen.cli import view_main

    with pytest.raises(SystemExit):
        view_main(argv)
    assert message in capsys.readouterr().err


def test_a_replay_is_refused_on_other_anatomy(tmp_path, capsys):
    from lumen.cli import view_main
    from lumen.viewer.scene import fingerprint

    frames = [{"step": 0, "nodes": [[0, 0, 0], [0, 0, 1]], "action": [0, 0], "left_mm": 1.0,
               "overlap_mm": 0.0, "off_route": False}]
    rep = {"version": record.REPLAY_VERSION, "anatomy": "arch", "anatomy_id":
           fingerprint(resolve("arch").asset), "target": None, "frames": frames,
           "route": [[0, 0, 0], [0, 0, 1]], "outcome": {}}
    path = record.save(rep, tmp_path / "r.json")
    view_main(["arch", "--replay", str(path), "--out", str(tmp_path / "ok.html")])
    assert (tmp_path / "ok.html").exists()
    with pytest.raises(SystemExit):
        view_main(["tortuous", "--replay", str(path), "--out", str(tmp_path / "no.html")])
    assert "recorded on different anatomy (arch)" in capsys.readouterr().err


def test_the_fingerprint_follows_the_geometry_not_the_object():
    from lumen.viewer.scene import fingerprint

    a, b = resolve("arch").asset, resolve("arch").asset
    assert fingerprint(a) == fingerprint(b) != fingerprint(resolve("tortuous").asset)


def test_a_title_cannot_take_the_place_of_the_data():
    scene = scene_dict(resolve("arch"))
    scene["label"] = "x __DATA__ __TITLE__ y"
    html = page.render_page(scene)
    head, _, body = html.partition("</title>")
    assert head.endswith("x __DATA__ __TITLE__ y")
    assert '"label":"x __DATA__ __TITLE__ y"' in body and '"mesh":{' in body


def test_scene_provenance_names_the_file_without_its_folder(tmp_path):
    from lumen.assets import procedural

    p = tmp_path / "deep" / "mine.json"
    p.parent.mkdir()
    procedural.aortic_arch_tree().save(str(p))
    s = scene_dict(resolve(str(p)))
    assert str(tmp_path) not in json.dumps(s) and s["source"] == {"kind": "asset", "name": "mine.json"}


def test_the_c_arm_draws_a_closed_surface():
    # A scan surface is open where the scan cut the vessel; the C-arm image adds path length
    # through a surface and needs a closed one, so it draws the lumen model instead.
    for cid in cases.names():
        s = scene_dict(resolve(cid))
        if s["mesh"]["kind"] != "scan surface":
            continue
        import base64

        v = np.frombuffer(base64.b64decode(s["carm_mesh"]["positions"]), np.float32).reshape(-1, 3)
        f = np.frombuffer(base64.b64decode(s["carm_mesh"]["indices"]), np.uint32).reshape(-1, 3)
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert (counts == 2).all(), cid                        # every edge shared by two faces
        assert _signed_volume(v, f) > 0
    assert scene_dict(resolve("arch"))["carm_mesh"] is None     # the model itself is used
