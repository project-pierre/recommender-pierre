# Always prefer setuptools over distutils
import os

from setuptools import setup, find_packages, Extension

# To use a consistent encoding
from codecs import open
from os import path
from setuptools import dist  # Install numpy right now

# dist.Distribution().fetch_build_eggs(["numpy"])

try:
    import numpy as np
except ImportError:
    exit("Please install numpy>=1.17.3 first.")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

# Get the long description from README.md
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    install_requires = [line.strip() for line in f.read().split("\n")]

cmdclass = {}

# ext = ".pyx" if USE_CYTHON else ".c"
ext = ".py" if USE_CYTHON else ".c"

extensions = [
    Extension(
        name="recommender_pierre.autoencoders.DeppAutoEncModel",
        sources=["recommender_pierre/autoencoders/DeppAutoEncModel" + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="recommender_pierre.autoencoders.CDAEModel",
        sources=["recommender_pierre/autoencoders/CDAEModel" + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="recommender_pierre.autoencoders.EASEModel",
        sources=["recommender_pierre/autoencoders/EASEModel" + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="recommender_pierre.bpr.BPRKNN",
        sources=["recommender_pierre/bpr/BPRKNN" + ext],
        include_dirs=[np.get_include()]
    ),
]

EXCLUDE_FILES = [
    "recommender_pierre/__init__.py",
    "recommender_pierre/autoencoders/__init__.py",
    "recommender_pierre/bpr/__init__.py"
]


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths


if USE_CYTHON:
    # See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
    extensions = cythonize(
        extensions,
        # get_ext_paths('recommender_pierre', EXCLUDE_FILES),
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )
    cmdclass.update({"build_ext": build_ext})

# This call to setup() does all the work
setup(
    name="recommender_pierre",
    version=__version__,
    description="Recommender-Pierre is a Scientific ToolKit for Collaborative Recommender Algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/project-pierre/recommender-pierre",
    author="Diego Correa da Silva",
    author_email="diegocorrea.cc@gmail.com",
    license="MIT",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License :: 2.0',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    keywords=(
        "Collaborative Filtering, Recommender Systems"
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=cythonize(extensions),
)
