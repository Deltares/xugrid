from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="xugrid",
    description="Xarray extension for 2D unstructured grids",
    long_description=long_description,
    url="https://github.com/deltares/xugrid",
    author="Huite Bootsma",
    author_email="huite.bootsma@deltares.nl",
    license="MIT",
    packages=find_packages(),
    package_dir={"xugrid": "xugrid"},
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    install_requires=["matplotlib", "xarray>=0.11"],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "pytest-cov",
            "sphinx",
            "sphinx_rtd_theme",
        ],
        "optional": [
            "vtk>=9.0",
            "pyvista",
        ],
    },
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="ugrid unstructured grid mesh",
)
