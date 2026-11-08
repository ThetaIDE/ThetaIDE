"""Explicit distribution boundary for authored and managed extension content."""
from setuptools import setup
from setuptools.command.build_py import build_py

# Compatibility modules are the only files distributed from the authoring namespace.
_COMPATIBILITY = {
    "src.usr": {"__init__"},
    "src.usr.algorithms": {"__init__", "agent_registry", "base_agent"},
    "src.usr.models": {"__init__"},
    "src.usr.models.sequential_spec": {"__init__", "sequential_spec"},
}


class FrameworkBuild(build_py):
    def find_data_files(self, package, src_dir):
        from pathlib import Path
        return [file for file in super().find_data_files(package, src_dir)
                if not any(part in ('node_modules','__pycache__','.git') for part in Path(file).parts)]

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package.startswith("src.usr"):
            allowed = _COMPATIBILITY.get(package, set())
            return [module for module in modules if module[1] in allowed]
        if package == "plot":
            return [module for module in modules if module[1] in {"__init__", "base", "capture", "manager"}]
        if package.startswith("frontend.plugins."):
            allowed = {"frontend.plugins.core", *{f"frontend.plugins.core.{name}" for name in
                       ("text_editor", "model_builder", "tensorboard", "arxiv_papers", "documentation")}}
            if package not in allowed: return []
        return modules


setup(cmdclass={"build_py": FrameworkBuild})
