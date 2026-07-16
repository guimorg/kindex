"""Tests for opt-in Unity asset ingestion (issue #17)."""
from __future__ import annotations

import pytest

from kindex.adapters import code
from kindex.config import Config
from kindex.store import Store

_GUID = "0123456789abcdef0123456789abcdef"
_UNITY_YAML = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &123\nGameObject:\n"


def _make_unity_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "Assets").mkdir(parents=True)
    (repo / "Scripts").mkdir()
    (repo / "Assets" / "Player.prefab").write_text(_UNITY_YAML)
    (repo / "Assets" / "Player.prefab.meta").write_text(
        f"fileFormatVersion: 2\nguid: {_GUID}\n"
    )
    (repo / "Scripts" / "main.cs").write_text("class Main {}\n")
    (repo / "Scripts" / "main.cs.meta").write_text(
        "fileFormatVersion: 2\nguid: fedcba9876543210fedcba9876543210\n"
    )
    return repo


@pytest.fixture
def local_only(monkeypatch):
    monkeypatch.setattr(code, "_run_ctags", lambda *_args: [])
    monkeypatch.setattr(code, "_check_cscope", lambda: False)
    monkeypatch.setattr(code, "_check_treesitter", lambda _lang: None)


@pytest.fixture
def store(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "kindex"))
    s = Store(cfg)
    yield s
    s.close()


def _modules(store):
    return {
        node["title"]: node
        for node in store.all_nodes(node_type="artifact", limit=100)
    }


def test_unity_files_excluded_by_default(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    result = code.ingest_code(store, repo)
    assert result.errors == []
    assert set(_modules(store)) == {"Scripts/main.cs"}


def test_unity_opt_in_creates_asset_modules_with_guids(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []

    modules = _modules(store)
    assert set(modules) == {"Assets/Player.prefab", "Scripts/main.cs"}
    assert not any(title.endswith(".meta") for title in modules)

    prefab = modules["Assets/Player.prefab"]["extra"]
    assert prefab["language"] == "Unity Prefab"
    assert prefab["serialization"] == "text"
    assert prefab["unity_guid"] == _GUID

    # Code files are also referenced by GUID in Unity, so scripts get one too
    script = modules["Scripts/main.cs"]["extra"]
    assert script["unity_guid"] == "fedcba9876543210fedcba9876543210"
    assert "serialization" not in script


def test_unity_binary_asset_is_sniffed_not_parsed(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Assets").mkdir(parents=True)
    (repo / "Assets" / "Terrain.asset").write_bytes(b"\x00\x01\x02binarystuff")

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []

    terrain = _modules(store)["Assets/Terrain.asset"]["extra"]
    assert terrain["serialization"] == "binary"
    assert terrain["line_count"] == 0


def test_unity_excludes_library_and_temp(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Library").mkdir(parents=True)
    (repo / "Temp").mkdir()
    (repo / "Assets").mkdir()
    (repo / "Library" / "x.asset").write_text(_UNITY_YAML)
    (repo / "Temp" / "y.asset").write_text(_UNITY_YAML)
    (repo / "Assets" / "keep.asset").write_text(_UNITY_YAML)

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []
    assert set(_modules(store)) == {"Assets/keep.asset"}


def test_include_extensions_generic(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Water.shader").write_text("Shader \"Custom/Water\" {}\n")

    result = code.ingest_code(
        store, repo, extra_extensions={".shader": "Unity Shader"},
    )
    assert result.errors == []
    water = _modules(store)["Water.shader"]["extra"]
    assert water["language"] == "Unity Shader"


def test_adapter_reads_code_ingest_config(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    cfg = Config(
        data_dir=str(tmp_path / "kindex"),
        code_ingest={"unity": True},
    )

    result = code.adapter.ingest(store, directory=str(repo), _config=cfg)
    assert result.errors == []
    assert "Assets/Player.prefab" in _modules(store)


def test_unity_excludes_nested_library_but_not_asset_folders(tmp_path, local_only, store):
    repo = tmp_path / "repo"
    (repo / "Games" / "ProjA" / "Library").mkdir(parents=True)
    (repo / "Games" / "ProjA" / "Assets").mkdir()
    (repo / "Assets" / "MyLibrary").mkdir(parents=True)
    (repo / "Games" / "ProjA" / "Library" / "x.asset").write_text(_UNITY_YAML)
    (repo / "Games" / "ProjA" / "Assets" / "keep.asset").write_text(_UNITY_YAML)
    (repo / "Assets" / "MyLibrary" / "thing.asset").write_text(_UNITY_YAML)

    result = code.ingest_code(store, repo, unity=True)
    assert result.errors == []
    assert set(_modules(store)) == {
        "Games/ProjA/Assets/keep.asset",
        "Assets/MyLibrary/thing.asset",
    }


def test_target_directory_kin_config_enables_unity(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    kin_dir = repo / ".kin"
    kin_dir.mkdir()
    (kin_dir / "config").write_text("code_ingest:\n  unity: true\n")

    result = code.adapter.ingest(store, directory=str(repo))
    assert result.errors == []
    assert "Assets/Player.prefab" in _modules(store)


def test_unity_guid_refresh_on_unchanged_asset(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    first = code.ingest_code(store, repo, unity=True)
    assert first.errors == []

    new_guid = "ffffffffffffffffffffffffffffffff"
    (repo / "Assets" / "Player.prefab.meta").write_text(
        f"fileFormatVersion: 2\nguid: {new_guid}\n"
    )

    second = code.ingest_code(store, repo, unity=True)
    assert second.errors == []
    prefab = _modules(store)["Assets/Player.prefab"]["extra"]
    assert prefab["unity_guid"] == new_guid


def test_adapter_flag_overrides_config_off(tmp_path, local_only, store):
    repo = _make_unity_repo(tmp_path)
    cfg = Config(
        data_dir=str(tmp_path / "kindex"),
        code_ingest={"unity": True},
    )

    result = code.adapter.ingest(
        store, directory=str(repo), _config=cfg, unity=False,
    )
    assert result.errors == []
    assert "Assets/Player.prefab" not in _modules(store)
