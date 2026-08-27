import json
from pathlib import Path

import pytest


def test_anatomy_pack_manifest_is_procedural_and_apache_licensed():
    from lumen.assets import anatomy_pack_manifest

    manifest = anatomy_pack_manifest()
    assert manifest["version"] == "lumen-anatomy/1"
    assert manifest["license"] == "Apache-2.0"
    assert manifest["provenance"] == "procedural"
    assert len(manifest["cases"]) == 6
    assert len({case["case_id"] for case in manifest["cases"]}) == 6
    assert all(case["license"] == "Apache-2.0" for case in manifest["cases"])
    assert all(case["provenance"] == "procedural" for case in manifest["cases"])


def test_anatomy_registry_parameters_are_immutable():
    from lumen.assets import ANATOMY_PACK

    with pytest.raises(TypeError):
        ANATOMY_PACK[0].parameters["n"] = 1


def test_anatomy_pack_materializes_every_case_and_validates_metadata():
    from lumen.assets import materialize_anatomy_pack, validate_anatomy_pack

    manifest = validate_anatomy_pack()
    assets = materialize_anatomy_pack()
    assert set(assets) == {case["case_id"] for case in manifest["cases"]}
    assert all(asset.provenance == "procedural" for asset in assets.values())
    assert all(asset.edges and asset.nodes for asset in assets.values())


def test_anatomy_pack_lookup_rejects_unknown_case():
    from lumen.assets import get_anatomy

    with pytest.raises(KeyError, match="unknown anatomy case"):
        get_anatomy("missing-case")


def test_notice_records_anatomy_pack_license_boundary():
    notice = (Path(__file__).resolve().parents[1] / "NOTICE").read_text()
    assert "Procedural anatomy pack policy:" in notice
    assert "provenance=\"procedural\"" in notice
    assert "unclear-license assets" in notice


def test_anatomy_cli_validates_and_reports_one_case(capsys):
    from lumen.cli import anatomy_main

    anatomy_main(["tree_bifurcation", "--validate"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["pack"] == "lumen-anatomy/1"
    assert payload["case"]["license"] == "Apache-2.0"
    assert payload["materialized"]["provenance"] == "procedural"
    assert payload["materialized"]["edges"] == 3
