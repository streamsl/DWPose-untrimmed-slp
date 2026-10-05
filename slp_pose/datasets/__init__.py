"""Dataset registry: built-in datasets, installed plugins and dataset files.

A dataset is ONE Python file with one Dataset subclass (base.py; README 'Adding a dataset';
examples/my_dataset.py), including any signer rules of its own (Dataset.rules, D22;
examples/custom_rules.py; a built-in dataset's own rules also resolve by name alone, builtin_rule).
`resolve(name_or_path, data_root)` finds it, in this order:
1. a path to a .py file: imported by path; it must define, or @register, exactly one Dataset subclass;
2. 'package.module:ClassName': that class, after importing the module (what `reference` gives a
   spawned GPU worker for a dataset that is not a file);
3. a registered name: the built-in modules in BUILTIN and every class decorated with @register;
4. an installed plugin: an entry point of the group ENTRY_POINT_GROUP named after the dataset,
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
from typing import Dict, List, Optional, Type, TypeVar, Union

from ..rules import Rule, framework_names
from ..select import FrameRule
from ..signer import PosedPeople, SignerChoice, VideoRule
from .base import (INDEX_COLUMNS, Dataset, DatasetError, Video, check_dataset_class, index_table, load_videos,
                   safe_id, schedule)

__all__ = ['BUILTIN', 'ENTRY_POINT_GROUP', 'INDEX_COLUMNS', 'Dataset', 'DatasetError', 'FrameRule', 'PosedPeople',
           'SignerChoice', 'Video', 'VideoRule', 'builtin_rule', 'dataset_class', 'index_table', 'load_videos', 'names',
           'reference', 'register', 'resolve', 'safe_id', 'schedule']

ENTRY_POINT_GROUP = 'slp_pose.datasets'
BUILTIN = ('bobsl', 'auslan_news')   # modules slp_pose.datasets.<name>, each registers its dataset <name>
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
    """The Dataset subclass for a name, a .py file path or 'module:ClassName' (module docstring)."""
    _load_builtins()
    text = os.fspath(name_or_path)
    if text.endswith('.py'):
        path = Path(text).expanduser()
        if not path.is_file():
            raise DatasetError(f'no such dataset file {path}')
        return _single_class(_import_file(path), str(path))
    if ':' in text:
        return _class_at(text)
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


def reference(dataset: Union[Dataset, Type[Dataset]]) -> Optional[str]:
    """What dataset_class needs to find this dataset's class in another process (a spawned GPU
    worker resolving the dataset's own per-frame rule, worker.task_rule): the resolved path of its
    dataset file, else 'module:ClassName'; None for a class that cannot be imported by name (one
    defined inside a function, made by type() or in a __main__ script)."""
    cls = dataset if isinstance(dataset, type) else type(dataset)
    for path, module in _FILES.items():
        if module.__name__ == cls.__module__:
            return str(path)
    found: object = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split('.'):
        found = getattr(found, part, None)
    if found is not cls or cls.__module__ in ('__main__', '__mp_main__'):
        return None
    return f'{cls.__module__}:{cls.__qualname__}'


def builtin_rule(name: str) -> Rule:
    """The own rule `name` of the first built-in dataset (BUILTIN) that has one: how a rule name
    given without its dataset resolves when it is not the framework's (rules.as_rule,
    worker.task_rule). Records, task pickles and scripts from before D22 name Auslan News' rules
    like the framework's. KeyError for another name."""
    own: Dict[str, Rule] = {}
    for module in BUILTIN:
        for rule in dataset_class(module).rules:
            own.setdefault(rule.name, rule)
    if name not in own:
        raise KeyError(f"unknown rule {name!r}; the framework's rules: {framework_names()}, the built-in datasets' "
                       f"own: {sorted(own)} (another dataset's own rules resolve through its Dataset.rule)")
    return own[name]


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


def _class_at(text: str) -> Type[Dataset]:
    """The class named by 'package.module:ClassName' (reference), checked."""
    module_name, _, qualname = text.partition(':')
    try:
        obj: object = importlib.import_module(module_name)
        for part in qualname.split('.'):
            obj = getattr(obj, part)
    except (ImportError, AttributeError, ValueError) as exc:
        raise DatasetError(f'cannot find the dataset class {text!r}: {exc}') from exc
    check_dataset_class(obj)   # raises DatasetError for anything but a Dataset subclass
    return obj


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
