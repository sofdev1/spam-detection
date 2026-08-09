"""
setup.py — makes `src` pip-installable as a local package (`pip install -e .`)
so `from src.xxx import yyy` works the same in notebooks, scripts, and
tests without sys.path hacks.
"""
from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str):
    with open(file_path) as f:
        requirements = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="spam_detection",
    version="0.0.1",
    author="PW Skills Data Science Team",
    packages=find_packages(),
    install_requires=[],  # kept empty; use requirements.txt directly
)
