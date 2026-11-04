"""Final BOLT public API and removal of retired Gaussian entry points."""
from pathlib import Path
import importlib.util
import mlbricks

def test_no_retired_python_entry_points():
    assert importlib.util.find_spec('mlbricks.gaussian') is None
    assert not hasattr(mlbricks, 'Gaussian')
    assert not hasattr(mlbricks, 'GaussianConfig')
    assert not hasattr(mlbricks.cache, 'GaussCache')
    assert all(hasattr(mlbricks, name) for name in mlbricks.__all__)

def test_source_is_bolt_owned():
    import mlbricks.bolt.model as model
    import mlbricks.bolt.config as config
    assert Path(model.__file__).name == 'model.py'
    assert Path(config.__file__).name == 'config.py'
