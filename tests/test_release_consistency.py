from __future__ import annotations

import inspect
from pathlib import Path
import re
import tomllib

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
    text = (ROOT / "mlbricks" / "model.py").read_text(encoding="utf-8")
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
