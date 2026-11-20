"""Dependency inspection and automated installation for Theta-IDE plugins and Hub components."""
from __future__ import annotations

import importlib.util
from importlib import metadata
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from packaging.requirements import InvalidRequirement, Requirement

# Mapping of PyPI package names to their canonical Python import module names
PACKAGE_TO_MODULE_MAP = {
    "gymnasium": "gymnasium",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "attrs": "attr",
    "protobuf": "google.protobuf",
    "tensorflow": "tensorflow",
}


def parse_package_name(requirement: str) -> Tuple[str, str]:
    """Parse requirement specifier to extract base package name and import module name.

    Examples:
      'gymnasium[atari]>=0.29.0' -> ('gymnasium', 'gymnasium')
      'ale-py>=0.8.1'           -> ('ale-py', 'ale_py')
      'scikit-learn'            -> ('scikit-learn', 'sklearn')
    """
    cleaned = requirement.strip()
    # Strip comments
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0].strip()

    # Match package name and optional extras (e.g., gymnasium[atari]>=1.0.0)
    match = re.match(r"^([A-Za-z0-9_\-\.]+)(?:\[([A-Za-z0-9_,\-\.]+)\])?", cleaned)
    if not match:
        return cleaned, cleaned.replace("-", "_")

    pkg_name = match.group(1).lower()
    mod_name = PACKAGE_TO_MODULE_MAP.get(pkg_name, pkg_name.replace("-", "_"))
    return pkg_name, mod_name


def is_module_installed(module_name: str) -> bool:
    """Check if a Python module is importable in the current environment."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _requirement_is_installed(requirement: Requirement, checked: set[str]) -> bool:
    # Cyclic extras can refer back to the distribution currently being checked.
    key = str(requirement)
    if key in checked:
        return True
    try:
        distribution = metadata.distribution(requirement.name)
    except metadata.PackageNotFoundError:
        # Retain support for bare import names (including standard-library modules).
        return not (requirement.specifier or requirement.extras or requirement.url) and is_module_installed(
            parse_package_name(requirement.name)[1])
    if requirement.url or not requirement.specifier.contains(distribution.version, prereleases=True):
        return False
    if not requirement.extras:
        return True
    checked.add(key)
    for value in distribution.requires or ():
        dependency = Requirement(value)
        if dependency.marker and any(dependency.marker.evaluate({"extra": extra}) for extra in requirement.extras):
            if not _requirement_is_installed(dependency, checked):
                return False
    return True


def check_missing_dependencies(requirements: List[str]) -> List[str]:
    """Return requirements that the current interpreter does not satisfy."""
    missing = []
    for req in requirements:
        candidate = re.split(r"\s+#", req.strip(), maxsplit=1)[0]
        if not candidate or candidate.startswith("#"):
            continue
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement:
            # Let pip handle requirements-file syntax it supports.
            missing.append(candidate)
            continue
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if not _requirement_is_installed(requirement, set()):
            missing.append(candidate)
    return missing


def load_requirements_from_file(req_file: Path) -> List[str]:
    """Read requirement lines from a requirements.txt file."""
    if not req_file.exists():
        return []
    try:
        lines = req_file.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    except Exception:
        return []


def install_packages(
    packages: List[str],
    python_executable: str | None = None,
    timeout_seconds: int = 300,
) -> Tuple[bool, str]:
    """Install packages via pip in the targeted Python interpreter.

    Args:
        packages: List of requirement strings (e.g. ['ale-py>=0.8.1', 'gymnasium']).
        python_executable: Python binary to run pip under. Defaults to sys.executable.
        timeout_seconds: Maximum execution time before timeout.

    Returns:
        (success: bool, message: str)
    """
    if not packages:
        return True, "No packages to install."

    python_bin = python_executable or sys.executable
    cmd = [python_bin, "-m", "pip", "install", *packages]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if res.returncode == 0:
            return True, f"Successfully installed: {', '.join(packages)}"
        return False, f"pip install failed (code {res.returncode}):\n{res.stdout}"
    except subprocess.TimeoutExpired:
        return False, f"Installation timed out after {timeout_seconds} seconds."
    except Exception as exc:
        return False, f"Failed to execute pip install: {exc}"
