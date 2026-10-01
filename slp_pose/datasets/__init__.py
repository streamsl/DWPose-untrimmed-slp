"""Dataset registry: built-in datasets, installed plugins and dataset files.

A dataset is ONE Python file with one Dataset subclass (base.py; README 'Adding a dataset';
examples/my_dataset.py). `resolve(name_or_path, data_root)` finds it, in this order:
1. a path to a .py file: imported by path; it must define, or @register, exactly one Dataset subclass;
2. a registered name: the built-in modules in BUILTIN and every class decorated with @register;
3. an installed plugin: an entry point of the group ENTRY_POINT_GROUP named after the dataset,
   pointing at a Dataset subclass ('pkg.module:MyDataset') or at a module defining exactly one, e.g.
   in the plugin's pyproject.toml:  [project.entry-points."slp_pose.datasets"]  my_data = "my_pkg.my_data"
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Type, TypeVar, Union

from .base import (INDEX_COLUMNS, Dataset, DatasetError, Video, check_dataset_class, index_table, load_videos,
                   safe_id, schedule)

__all__ = ['BUILTIN', 'ENTRY_POINT_GROUP', 'INDEX_COLUMNS', 'Dataset', 'DatasetError', 'Video', 'dataset_class',
           'index_table', 'load_videos', 'names', 'register', 'resolve', 'safe_id', 'schedule']

ENTRY_POINT_GROUP = 'slp_pose.datasets'
BUILTIN = ('bobsl', 'auslan_news')   # modules slp_pose.datasets.<name>, each registers its dataset
D = TypeVar('D', bound=Type[Dataset])
_REGISTRY: Dict[str, Type[Dataset]] = {}
_FILES: Dict[Path, ModuleType] = {}   # dataset files imported so far, by resolved path


def register(cls: D) -> D:
    """Class decorator: check `cls` (base.check_dataset_class) and register it under cls.name."""
    check_dataset_class(cls)
    taken = _REGISTRY.get(cls.name)
    if taken is not None and taken is not cls:
        raise DatasetError(f'dataset name {cls.name!r} is already taken by {taken.__module__}.{taken.__qualname__}')
    _REGISTRY[cls.name] = cls
    return cls


def names() -> List[str]:
    """Every dataset name known without a file path: built-in, registered and installed plugins."""
    _load_builtins()
    return sorted(set(_REGISTRY) | set(_entry_points()))


def dataset_class(name_or_path: Union[str, os.PathLike]) -> Type[Dataset]:
    """The Dataset subclass for a name or a .py file path (module docstring)."""
    _load_builtins()
    text = os.fspath(name_or_path)
    if text.endswith('.py'):
        path = Path(text).expanduser()
        if not path.is_file():
            raise DatasetError(f'no such dataset file {path}')
        return _single_class(_import_file(path), str(path))
    if text in _REGISTRY:
        return _REGISTRY[text]
    entry = _entry_points().get(text)
    if entry is None:
        raise DatasetError(f'unknown dataset {text!r}; known: {", ".join(names())} (or pass a .py dataset file)')
    obj = entry.load()
    cls = _single_class(obj, f'entry point {entry.value}') if isinstance(obj, ModuleType) else obj
    check_dataset_class(cls)
    if cls.name != text:
        raise DatasetError(f'entry point {text!r} ({entry.value}) defines the dataset {cls.name!r}')
    return cls


def resolve(name_or_path: Union[str, os.PathLike], data_root: os.PathLike) -> Dataset:
    """The dataset `name_or_path` (dataset_class) constructed on `data_root`."""
    return dataset_class(name_or_path)(Path(data_root))


def _load_builtins() -> None:
    for module in BUILTIN:
        importlib.import_module(f'{__name__}.{module}')


def _entry_points() -> Dict[str, object]:
    """{name: EntryPoint} of ENTRY_POINT_GROUP (importlib.metadata of Python 3.8 and later)."""
    from importlib import metadata
    found = metadata.entry_points()
    group = found.select(group=ENTRY_POINT_GROUP) if hasattr(found, 'select') else found.get(ENTRY_POINT_GROUP, ())
    return {entry.name: entry for entry in group}


def _import_file(path: Path) -> ModuleType:
    """Import a dataset file under a unique module name (kept in sys.modules, as dataclasses and
    pickle look classes up there); each file is imported once."""
    path = path.resolve()
    if path in _FILES:
        return _FILES[path]
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    name = f'slp_pose_dataset_{re.sub(r"[^0-9A-Za-z_]", "_", path.stem)}_{digest}'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DatasetError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    _FILES[path] = module
    return module


def _single_class(module: ModuleType, origin: str) -> Type[Dataset]:
    """The one Dataset subclass that `module` registers, or else the one it defines."""
    registered = [cls for cls in _REGISTRY.values() if cls.__module__ == module.__name__]
    defined = [obj for obj in vars(module).values() if isinstance(obj, type) and issubclass(obj, Dataset)
               and obj is not Dataset and obj.__module__ == module.__name__]
    found = registered or defined
    if len(found) != 1:
        listed = ', '.join(cls.__qualname__ for cls in found) or 'none'
        raise DatasetError(f'{origin} must define (or @register) exactly one Dataset subclass; found {listed}')
    check_dataset_class(found[0])
    return found[0]
