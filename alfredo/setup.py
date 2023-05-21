from setuptools import find_packages, setup

exec(open("version.py").read())

setup(
    name="alfredo",
    version=__version__,
    url="https://github.com/tartavull/alfredo.git",
    author="Alfredo Contributors",
    author_email="tartavull@gmail.com",
    packages=["alfredo"],
    package_dir={"alfredo": "./"},
)
