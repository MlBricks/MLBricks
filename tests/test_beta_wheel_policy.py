from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _workflow() -> str:
    return (ROOT / ".github" / "workflows" / "native-wheels-beta.yml").read_text(
        encoding="utf-8"
    )


def test_beta_version_and_torch_abi_line_are_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "1.0.0b1"
    assert "torch>=2.10,<2.11" in project["dependencies"]


def test_source_install_defaults_native_compilation_off() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'return os.getenv(name, "0") == "1"' in setup
    assert "MLBRICKS_NATIVE_LINEINFO" in setup


def test_cuda_release_builds_fail_fast_instead_of_silently_dropping_extensions() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'MLBRICKS_EXPECT_CUDA_NATIVE' in setup
    assert 'MLBRICKS_EXPECT_CUDA_VERSION' in setup
    assert "installed PyTorch build is CPU-only" in setup
    assert "CUDA_HOME was not detected" in setup


def test_release_workflow_builds_fat_cuda_and_fallback_wheel() -> None:
    workflow = _workflow()
    assert 'TORCH_CUDA_ARCH_LIST: "7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"' in workflow
    assert "py3-none-any.whl" in workflow
    assert "linux-cuda" in workflow
    assert "windows-cuda" in workflow
    assert "macos-native" in workflow
    assert "gh-action-pypi-publish" in workflow


def test_native_no_isolation_builds_install_modern_setuptools() -> None:
    workflow = _workflow()
    assert workflow.count('"setuptools>=77.0.3"') >= 4
    assert "--no-isolation" in workflow


def test_linux_and_windows_use_official_pytorch_cuda_128_index() -> None:
    workflow = _workflow()
    assert 'PYTORCH_CUDA_INDEX: "https://download.pytorch.org/whl/cu128"' in workflow
    assert workflow.count("--index-url") >= 2
    assert workflow.count('MLBRICKS_EXPECT_CUDA_NATIVE: "1"') == 2
    assert workflow.count('MLBRICKS_EXPECT_CUDA_VERSION: "12.8"') == 2
    assert "torch.version.cuda.startswith('12.8')" in workflow
    assert "torch.version.cuda.startswith('12.8')" in workflow or "torch.version.cuda.startswith(\"12.8\")" in workflow


def test_windows_wheel_verifies_all_six_native_extensions_before_upload() -> None:
    workflow = _workflow()
    assert "All six Windows native extensions packaged: PASS" in workflow
    for ext in (
        "mlbricks/_C",
        "mlbricks/_gauss_cuda",
        "mlbricks/_vision_native",
        "mlbricks/ffnbrick/_C",
        "mlbricks/residualbrick/_C",
        "mlbricks/elasticbit/_C",
    ):
        assert ext in workflow


def test_macos_beta_targets_apple_silicon_only() -> None:
    workflow = _workflow()
    assert "runs-on: macos-14" in workflow
    assert "macos-15-intel" not in workflow
    assert "macOS ARM64 native" in workflow
    assert 'MACOSX_DEPLOYMENT_TARGET: "11.0"' in workflow
    assert "platform.machine() == 'arm64'" in workflow
    assert "macos-arm64-py" in workflow


def test_artifact_actions_use_node24_generation() -> None:
    workflow = _workflow()
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/download-artifact@v7" in workflow
    assert "actions/upload-artifact@v4" not in workflow
    assert "actions/download-artifact@v5" not in workflow
