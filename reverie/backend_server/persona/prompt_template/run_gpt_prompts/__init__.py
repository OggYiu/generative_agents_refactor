"""
run_gpt_prompts — auto-discovering package of prompt functions.

Every public (non-underscore) .py module in this directory is imported
and all its public names are re-exported.  To add a prompt function,
just drop a new .py file here — no manifest to maintain.
"""
import pkgutil
import importlib
import pathlib

_pkg_path = str(pathlib.Path(__file__).resolve().parent)
_all_names = []

for _importer, _modname, _ispkg in pkgutil.iter_modules([_pkg_path]):
    if _modname.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_modname}")
    for _name in getattr(_mod, "__all__", None) or dir(_mod):
        if _name.startswith("_"):
            continue
        globals()[_name] = getattr(_mod, _name)
        _all_names.append(_name)

__all__ = _all_names
