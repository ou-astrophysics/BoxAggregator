import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    install_requires=[
        "numba",
        "numpy",
        "pandas",
        "scipy",
        "FacilityLocation @ git+ssh://github.com/ou-escape-eco/FacilityLocation@master",
    ],
    dependency_links=["https://github.com/ou-escape-eco/FacilityLocation.git"],
)
