from __future__ import annotations

import inspect
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from mlbricks import ESA, ESAModel, ESAModelConfig


ROOT = Path(__file__).resolve().parents[1]


def test_all_public_compile_defaults_are_safe() -> None:
    assert inspect.signature(ESA.__init__).parameters["compile_mode"].default == "default"
    assert inspect.signature(ESA.compile).parameters["mode"].default == "default"
    assert ESAModelConfig(vocab_size=128).training_compile_mode == "default"
    assert inspect.signature(ESAModel.prefill).parameters["compile_mode"].default == "default"
    assert inspect.signature(ESAModel.compile_generation).parameters["mode"].default == "default"
    assert inspect.signature(ESAModel.generate).parameters["compile_mode"].default == "default"
    assert inspect.signature(ESAModel.generate).parameters["compile"].default is True


def test_package_description_matches_uniform_backend_release() -> None:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    description = project["description"].lower()
    assert "auto/native/pytorch" in description
    assert "gaussian" in description
    assert "bricks" in description
    assert "pulse" not in description
    assert "flare" not in description


def test_readme_has_no_stale_compile_default_claims() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    stale_patterns = (
        r"training_compile_mode\s+reduce-overhead",
        r"(?:Optional|Default)\s+training\s+compilation:\*\*"
        r".*?mode=[\"'`]reduce-overhead",
    )
    for pattern in stale_patterns:
        assert re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ) is None

    assert 'training_compile_mode="default"' in text
    assert 'compile_mode="default"' in text


def test_source_docs_match_compile_policy() -> None:
    text = (ROOT / "mlbricks" / "esa" / "model.py").read_text(encoding="utf-8")
    assert "fixed-shape ESA-Lightning decode step" in text
    assert "default" in text


def test_no_real_git_conflict_markers() -> None:
    conflict = re.compile(r"^(?:<<<<<<< .+|=======|>>>>>>> .+)$")
    for path in (ROOT / "mlbricks").rglob("*.py"):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            assert conflict.fullmatch(line) is None, (
                f"Conflict marker in {path.relative_to(ROOT)}:{line_number}"
            )


def test_pypi_release_metadata_and_license_files_are_consistent() -> None:
    import mlbricks

    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == mlbricks.__version__ == "1.0.0"
    assert project["license"] == "PolyForm-Noncommercial-1.0.0"
    for relative in project["license-files"]:
        assert (ROOT / relative).is_file(), f"Missing release license file: {relative}"
    assert not (ROOT / "PKG-INFO").exists(), "PKG-INFO is generated and must not be committed"


def test_readme_documents_every_native_build_flag() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    flags = (
        "MLBRICKS_BUILD_CORE_NATIVE",
        "MLBRICKS_BUILD_BOLT_NATIVE",
        "MLBRICKS_BUILD_VESA_NATIVE",
        "MLBRICKS_BUILD_VISION_NATIVE",
        "MLBRICKS_BUILD_FFNBRICK_NATIVE",
        "MLBRICKS_BUILD_RESIDUALBRICK_NATIVE",
        "MLBRICKS_BUILD_ELASTICBIT_NATIVE",
    )
    for flag in flags:
        assert flag in setup
        assert f"{flag}=0" in readme


def test_sdist_manifest_contains_linked_api_and_soup_license() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "include API.md" in manifest
    assert "mlbricks/soup/LICENSE_SOUP.txt" in readme
