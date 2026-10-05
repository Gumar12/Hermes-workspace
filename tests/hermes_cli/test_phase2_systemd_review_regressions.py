import subprocess

import pytest

from gateway import service_identity, systemd_runtime, systemd_unit_render
from hermes_cli import gateway as gateway_cli


@pytest.mark.parametrize(
    "manager_output",
    [
        "Environment=HERMES_HOME=/srv/hermes PATH=/usr/bin\n",
        "Environment=PATH=/usr/bin HERMES_HOME=/srv/hermes\n",
    ],
)
def test_sync_home_from_manager_environment_ignores_property_wrapper(
    monkeypatch, tmp_path, manager_output
):
    monkeypatch.setattr(systemd_runtime, "unit_path", lambda system=False: tmp_path / "missing.service")
    monkeypatch.setattr(
        systemd_runtime,
        "run_systemctl",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["systemctl"], returncode=0, stdout=manager_output, stderr=""
        ),
    )
    monkeypatch.setenv("HERMES_HOME", "/root/.hermes")

    systemd_runtime.sync_home_from_unit(system=True)

    assert systemd_runtime.os.environ["HERMES_HOME"] == "/srv/hermes"


def test_sync_home_prefers_unit_pin_without_manager_fallback(monkeypatch, tmp_path):
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(
        systemd_unit_render.systemd_env_line("HERMES_HOME", "/unit/hermes"),
        encoding="utf-8",
    )
    monkeypatch.setattr(systemd_runtime, "unit_path", lambda system=False: unit)

    def unexpected_manager_query(*args, **kwargs):
        raise AssertionError("manager fallback should not run when the unit pins HERMES_HOME")

    monkeypatch.setattr(systemd_runtime, "run_systemctl", unexpected_manager_query)
    monkeypatch.setenv("HERMES_HOME", "/caller/hermes")

    systemd_runtime.sync_home_from_unit(system=True)

    assert systemd_runtime.os.environ["HERMES_HOME"] == "/unit/hermes"


@pytest.mark.parametrize("name", ["HERMES_HOME", "LD_LIBRARY_PATH"])
@pytest.mark.parametrize(
    "value",
    [
        "/plain/path",
        "/opt/pct%dir/lib",
        r"/opt/a\b/lib",
        '/opt/cu"da/lib',
        r'/opt/a\b/"quoted"/pct%dir',
    ],
)
def test_unit_environment_reader_round_trips_writer(tmp_path, name, value):
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(systemd_unit_render.systemd_env_line(name, value), encoding="utf-8")

    assert service_identity.unit_environment_value(unit, name) == value


def test_installed_ld_library_path_round_trip_is_byte_stable(monkeypatch, tmp_path):
    value = '/opt/a\\b/lib:/opt/"cuda"%/lib'
    unit = tmp_path / "hermes-gateway.service"
    encoded = systemd_unit_render.systemd_env_line("LD_LIBRARY_PATH", value)
    unit.write_text(encoded, encoding="utf-8")
    monkeypatch.setattr(
        systemd_unit_render.systemd_identity,
        "unit_path",
        lambda system=False: unit,
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    first = systemd_unit_render.ld_library_path_line(False)
    second = systemd_unit_render.ld_library_path_line(False)

    assert first == encoded
    assert second == first


@pytest.mark.parametrize(("scope", "system"), [("user", False), ("system", True)])
@pytest.mark.parametrize("consent", [True, False])
def test_setup_preserves_legacy_service_preflight(
    monkeypatch, scope, system, consent
):
    events = []
    monkeypatch.setattr(gateway_cli, "prompt_linux_gateway_install_scope", lambda: scope)
    monkeypatch.setattr(gateway_cli._systemd_legacy, "has_units", lambda: True)
    monkeypatch.setattr(
        gateway_cli, "print_legacy_unit_warning", lambda: events.append("warning")
    )
    monkeypatch.setattr(
        gateway_cli, "prompt_yes_no", lambda *args, **kwargs: consent
    )
    monkeypatch.setattr(
        gateway_cli,
        "remove_legacy_hermes_units",
        lambda **kwargs: events.append("remove"),
    )
    monkeypatch.setattr(
        gateway_cli._systemd_lifecycle,
        "install",
        lambda **kwargs: events.append(("install", kwargs)),
    )

    if system:
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(
            gateway_cli, "_default_system_service_user", lambda: "alice"
        )
    else:
        monkeypatch.setattr(
            gateway_cli, "refuses_container_user_scope_install", lambda **kwargs: False
        )

    assert gateway_cli.install_linux_gateway_from_setup() == (scope, True)

    install_index = next(i for i, event in enumerate(events) if isinstance(event, tuple))
    assert events[0] == "warning"
    if consent:
        assert events.index("remove") < install_index
    else:
        assert "remove" not in events

    install_kwargs = events[install_index][1]
    assert install_kwargs["system"] is system
    if system:
        assert install_kwargs["run_as_user"] == "alice"
