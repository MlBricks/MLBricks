from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_beta_version_and_torch_abi_line_are_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "1.0.0b1"
    assert "torch>=2.10,<2.11" in project["dependencies"]


def test_source_install_defaults_native_compilation_off() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'return os.getenv(name, "0") == "1"' in setup
    assert 'MLBRICKS_NATIVE_LINEINFO' in setup


def test_release_workflow_builds_fat_cuda_and_fallback_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "native-wheels-beta.yml").read_text(encoding="utf-8")
    assert 'TORCH_CUDA_ARCH_LIST: "7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"' in workflow
    assert "py3-none-any.whl" in workflow
    assert "linux-cuda" in workflow
    assert "windows-cuda" in workflow
    assert "macos-native" in workflow
    assert "gh-action-pypi-publish" in workflow
