from setuptools import setup, find_packages

setup(
    name='nvae',
    version='0.0.0',
    packages=find_packages(include=['src', 'src.*', 'data', 'data.*'])
)
