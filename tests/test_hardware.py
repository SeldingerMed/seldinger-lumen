"""Hardware health-report helpers."""

import sys
import types

from lumen import hardware


def _fake_warp(*, log_level=20, version=None, init=None, cuda_devices=0):
    config = types.SimpleNamespace(log_level=log_level)
    if version is not None:
        config.version = version
    return types.SimpleNamespace(
        LOG_DEBUG=10,
        LOG_INFO=20,
        LOG_WARNING=30,
        LOG_ERROR=40,
        config=config,
        init=(init or (lambda: None)),
        get_cuda_device_count=lambda: cuda_devices,
    )


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
    fake_wp = _fake_warp()
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.delenv(hardware.BACKEND_LOG_ENV, raising=False)

    hardware.configure_backend_logging()

    assert fake_wp.config.log_level == fake_wp.LOG_WARNING


def test_configure_backend_logging_honors_env_override(monkeypatch):
    fake_wp = _fake_warp(log_level=30)
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.setenv(hardware.BACKEND_LOG_ENV, "debug")

    hardware.configure_backend_logging()

    assert fake_wp.config.log_level == fake_wp.LOG_DEBUG


def test_configure_backend_logging_rejects_unknown_level(monkeypatch):
    fake_wp = _fake_warp(log_level=30)
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.setenv(hardware.BACKEND_LOG_ENV, "chatty")

    try:
        hardware.configure_backend_logging()
    except ValueError as e:
        assert hardware.BACKEND_LOG_ENV in str(e)
    else:
        raise AssertionError("unknown backend log level should fail")


def test_detect_device_honors_cpu_preference(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", _fake_warp())

    assert hardware.detect_device(prefer="cpu") == "cpu"


def test_detect_device_uses_cuda_when_warp_reports_visible_devices(monkeypatch):
    calls = []
    fake_wp = _fake_warp(init=lambda: calls.append("init"), cuda_devices=2)
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.delenv(hardware.BACKEND_LOG_ENV, raising=False)

    assert hardware.detect_device() == "cuda"
    assert calls == ["init"]


def test_detect_device_falls_back_to_cpu_when_warp_init_fails(monkeypatch):
    def fail_init():
        raise RuntimeError("backend unavailable")

    fake_wp = _fake_warp(init=fail_init, cuda_devices=1)
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.delenv(hardware.BACKEND_LOG_ENV, raising=False)

    assert hardware.detect_device(prefer="cuda") == "cpu"


def test_describe_reports_validated_backend_with_fake_warp_and_newton(monkeypatch):
    fake_wp = _fake_warp(version=hardware.VALIDATED_WARP_VERSION, cuda_devices=1)
    fake_newton = types.SimpleNamespace(
        __version__=hardware.VALIDATED_NEWTON_VERSION,
        __git_commit__=hardware.VALIDATED_NEWTON_REF,
    )
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    info = hardware.describe()

    assert info["device"] == "cuda"
    assert info["warp"] == hardware.VALIDATED_WARP_VERSION
    assert info["cuda_devices"] == 1
    assert info["newton"] == hardware.VALIDATED_NEWTON_VERSION
    assert info["newton_ref"] == hardware.VALIDATED_NEWTON_REF
    assert info["newton_available"] is True
    assert info["backend_validated"] is True
