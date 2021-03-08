import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=["BoxAggregator"],
    install_requires=[
        "numba",
        "numpy",
        "pandas",
        "scipy",
        "FacilityLocation @ git+ssh://git@github.com/ou-escape-eco/FacilityLocation@master",
    ],
)
