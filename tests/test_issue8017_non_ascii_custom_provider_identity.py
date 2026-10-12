"""Regression tests for #8017 — a non-ASCII custom provider name is not dropped.

Reported symptom: a ``custom_providers[]`` entry whose ``name`` contains no ASCII
characters (a pure-CJK name such as ``晨光鑫遇专用``) slugified to the empty
string, so the model-catalog builder treated it as "no provider" and skipped the
whole entry. Its models never appeared in the WebUI picker while the CLI kept
resolving the same endpoint, with no error or diagnostic on the response.

Root cause: ``_custom_provider_slug_from_name()`` slugifies with an ASCII-only
character class and returned ``""`` when nothing survived. The entry then lost
its identity in ``get_available_models()`` (the only place the catalog decides
which named groups exist), and the group was never built.

The fix keeps the name's own characters when no ASCII identifier character
survives, so the WebUI mints the same ``custom:<name>`` the Agent resolves it to
(``_agent_custom_provider_slug`` already folds spaces and keeps such characters).
A name with ANY ASCII identifier character is unchanged, so existing ASCII
identities are preserved.

That identity is only safe once three things follow from it, and all three are
pinned below: a fallback entry is admitted only when no pre-existing owner claims
the slug (``_custom_provider_identity_owners``), it never reads the shared
``CUSTOM_<SLUG>_API_KEY`` a different provider may own
(``_custom_provider_record_may_take_convention_key``), and the Save path does not
copy its connection onto the ``model:`` block, so the entry stays the route's
authority and keeps its own endpoint and credential.

The first test is the load-bearing assumption the rest of the shape rests on, and
it is red on master: there the entry minted no identity, so the resolver never
found it and returned nothing at all.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

import api.config as config
from api import routes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_models_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_models_cache_path", tmp_path / "models_cache.json")
    config.invalidate_models_cache()
    yield
    config.invalidate_models_cache()


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Keep the credential path off the real ~/.hermes (#8017 config uses key_env).

    The reported config resolves the key through ``key_env``, so the catalog read
    reaches the auth store; without an isolated home the run would touch the real
    ``~/.hermes/auth.json`` (hermes_cli guards against exactly that).
    """
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BASE_HOME", str(home))


def _stub_provider_modules(monkeypatch, detected_provider_ids: list[dict]):
    fake_models = types.ModuleType("hermes_cli.models")
    fake_models.list_available_providers = lambda: detected_provider_ids
    fake_auth = types.ModuleType("hermes_cli.auth")
    fake_auth.get_auth_status = lambda _pid: {"key_source": "config_yaml"}
    monkeypatch.setitem(sys.modules, "hermes_cli.models", fake_models)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
    monkeypatch.setattr(
        config,
        "_get_auth_store_path",
        lambda: config.Path("/tmp/does-not-exist-auth.json"),
    )


def _set_cfg(provider_name: str):
    """Swap in a config with one custom_providers entry, returning a restore fn.

    The mtime is pinned so get_available_models()'s mtime-guard does not reload
    the on-disk config over the patch (the same pattern the other catalog tests
    use — without it the real ~/.hermes config wins and the test is a no-op).
    """
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(
        {
            "model": {
                "default": "DeepSeek-V4.1-Flash",
                "provider": "custom",
                "base_url": "http://127.0.0.1:8317/v1",
                "api_key": "${HERMES_CUSTOM_127_0_0_1_8317_API_KEY}",
            },
            "custom_providers": [
                {
                    "name": provider_name,
                    "base_url": "http://127.0.0.1:8317/v1",
                    "key_env": "HERMES_CUSTOM_127_0_0_1_8317_API_KEY",
                    "model": "DeepSeek-V4.1-Flash",
                }
            ],
        }
    )
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()
    config.invalidate_models_cache()

    def restore():
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return restore


def _write_cfg(tmp_path, body: str) -> "config.Path":
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _load(cfg_path):
    loaded = config._load_yaml_config_file(cfg_path)
    config.cfg.clear()
    config.cfg.update(loaded)
    return loaded


@pytest.fixture(autouse=True)
def _restore_shared_config():
    """Undo the picker tests' in-place mutation of the shared ``config.cfg``.

    ``_load`` clears and refills the module-global config dict, which is the same
    object as ``config._cfg_cache`` (``cfg`` is an alias). Nothing restored it, so a
    later test could inherit this file's fake provider config and its result would
    depend on test order. Snapshot the dict and the mtime/path guards and put them
    back, so the fake config never escapes this module.
    """
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    yield
    config.cfg.clear()
    config.cfg.update(old_cfg)
    config._cfg_mtime = old_mtime
    config._cfg_path = old_path


def test_a_fallback_entry_picked_in_the_picker_is_served_from_its_own_connection(monkeypatch):
    """The assumption the whole change rests on (#8026 rescope, request 1).

    The Save path does NOT copy a fallback entry's connection onto the ``model:``
    block, so the block the picker leaves behind declares no connection of its own.
    The route must then be served from the ENTRY's own endpoint and credential.

    Asserted as the PAIR the resolver returns, because a wrong route with the right
    url is exactly the failure the report describes: the key is what went missing.
    """
    u = "http://entry.example/v1"
    cfg_shape = {
        "model": {"provider": "custom:晨光", "default": "chat-model"},
        "custom_providers": [
            {"name": "晨光", "base_url": u, "api_key": "sk-entry", "model": "chat-model"}
        ],
    }
    monkeypatch.setattr(config, "cfg", dict(cfg_shape))
    monkeypatch.setattr(config, "get_config", lambda: dict(cfg_shape))
    monkeypatch.delenv("CUSTOM_CUSTOM_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM__API_KEY", raising=False)

    assert config.resolve_custom_provider_connection("custom:晨光") == ("sk-entry", u), (
        "a fallback entry the picker selected is served from its own endpoint and key"
    )
