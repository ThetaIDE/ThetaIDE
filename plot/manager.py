"""Compatibility adapter for framework plotting infrastructure."""
from src.app.core.import_aliases import alias_module
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.app.core.plotting.manager", run_name="__main__")
else:
    alias_module(__name__, "src.app.core.plotting.manager")
