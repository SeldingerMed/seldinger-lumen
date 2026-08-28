import re
from importlib.metadata import version

import lumen


_SEMVER = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?"
)


def test_distribution_and_runtime_versions_are_semver_and_match():
    distribution_version = version("seldinger-lumen")
    assert _SEMVER.fullmatch(distribution_version)
    assert _SEMVER.fullmatch(lumen.__version__)
    assert distribution_version == lumen.__version__
