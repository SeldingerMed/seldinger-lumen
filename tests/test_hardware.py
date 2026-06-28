"""Hardware health-report helpers."""

import sys
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


def test_configure_backend_logging_defaults_to_warning(monkeypatch):
    fake_wp = types.SimpleNamespace(
        LOG_DEBUG=10,
        LOG_INFO=20,
        LOG_WARNING=30,
        LOG_ERROR=40,
        config=types.SimpleNamespace(log_level=20),
    )
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.delenv(hardware.BACKEND_LOG_ENV, raising=False)

    hardware.configure_backend_logging()

    assert fake_wp.config.log_level == fake_wp.LOG_WARNING


def test_configure_backend_logging_honors_env_override(monkeypatch):
    fake_wp = types.SimpleNamespace(
        LOG_DEBUG=10,
        LOG_INFO=20,
        LOG_WARNING=30,
        LOG_ERROR=40,
        config=types.SimpleNamespace(log_level=30),
    )
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.setenv(hardware.BACKEND_LOG_ENV, "debug")

    hardware.configure_backend_logging()

    assert fake_wp.config.log_level == fake_wp.LOG_DEBUG


def test_configure_backend_logging_rejects_unknown_level(monkeypatch):
    fake_wp = types.SimpleNamespace(
        LOG_DEBUG=10,
        LOG_INFO=20,
        LOG_WARNING=30,
        LOG_ERROR=40,
        config=types.SimpleNamespace(log_level=30),
    )
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.setenv(hardware.BACKEND_LOG_ENV, "chatty")

    try:
        hardware.configure_backend_logging()
    except ValueError as e:
        assert hardware.BACKEND_LOG_ENV in str(e)
    else:
        raise AssertionError("unknown backend log level should fail")
