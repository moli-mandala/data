"""Import a raw_data source package's flat sibling modules without leaking them.

Source packages under ``data/other/forms/raw_data/`` import their siblings by bare name
(``import layout``, ``import parse``, ``import import_source``). When several packages' tests
share one pytest session, the first package to be imported owns those names in ``sys.modules``
and on ``sys.path``, and later packages silently receive the wrong module. ``load`` imports a
package with only its own directory in front, records its modules, and then restores the
interpreter state; ``active`` re-installs exactly those modules (and the directory, for lazy
imports) for the duration of one test module.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path


class Package:
    def __init__(self, directory: Path, modules: dict):
        self.directory = str(directory)
        self.modules = modules

    @contextlib.contextmanager
    def active(self):
        saved = {name: sys.modules.get(name) for name in self.modules}
        sys.modules.update(self.modules)
        sys.path.insert(0, self.directory)
        try:
            yield
        finally:
            sys.path.remove(self.directory)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def load(directory: Path, *names: str) -> tuple[Package, list]:
    """Import ``names`` from ``directory``; returns the package handle and the modules."""
    local = sorted(p.stem for p in Path(directory).glob("*.py"))
    saved = {name: sys.modules.pop(name) for name in local if name in sys.modules}
    sys.path.insert(0, str(directory))
    try:
        modules = [importlib.import_module(name) for name in names]
        package = Package(directory, {n: sys.modules[n] for n in local if n in sys.modules})
    finally:
        sys.path.remove(str(directory))
        for name in local:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return package, modules
