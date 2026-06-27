"""Hardware health-report helpers."""

import types

from lumen import hardware


def test_newton_install_ref_reads_direct_url_metadata(monkeypatch):
    class FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return (
                '{"url":"https://github.com/newton-physics/newton.git",'
                '"vcs_info":{"commit_id":"abc123","vcs":"git"}}'
            )

    monkeypatch.setattr(hardware.metadata, "distribution", lambda name: FakeDist())

    assert hardware._newton_install_ref(types.SimpleNamespace(__version__="1.4.0.dev0")) == "abc123"


def test_newton_install_ref_prefers_module_commit_attributes(monkeypatch):
    def fail(_name):
        raise AssertionError("metadata should not be consulted when module exposes a commit")

    monkeypatch.setattr(hardware.metadata, "distribution", fail)

    assert hardware._newton_install_ref(types.SimpleNamespace(__git_commit__="def456")) == "def456"
