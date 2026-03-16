from __future__ import annotations

import sys
from importlib import import_module


def alias_module(target: str, alias_name: str):
    module = import_module(target)
    sys.modules[alias_name] = module
    return module
