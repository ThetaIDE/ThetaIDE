"""Explicit distribution boundary for authored and managed extension content."""
from setuptools import setup
from setuptools.command.build_py import build_py


class FrameworkBuild(build_py):
    def find_data_files(self, package, src_dir):
        from pathlib import Path
        return [file for file in super().find_data_files(package, src_dir)
                if not any(part in ('node_modules','__pycache__','.git') for part in Path(file).parts)]

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package.startswith("src.usr"):
            return []  # the authoring workspace is never distributed
        return modules


setup(cmdclass={"build_py": FrameworkBuild})
