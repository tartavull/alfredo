from setuptools import setup, find_packages
exec(open('version.py').read())

setup(
    name='genetic-intelligence',
    version=__version__,
    url='https://github.com/tartavull/genetic-intelligence.git',
    author='Genetic Intelligence Github Contributors',
    author_email='tartavull@gmail.com',
    description='Attempt to increase sample efficiency of reinforcement',
    packages=["genetic_intelligence"],    
    package_dir={"genetic_intelligence": "./"},
)
