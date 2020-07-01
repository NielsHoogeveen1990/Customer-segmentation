from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='navara',
    keywords='',
    version='1.0',
    author='Niels Hoogeveen',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    description='Unsupervised K-Means model to create customer segmentations for the energy market in the Netherlands.',
    entry_points={'console_scripts': ['navara = navara.cli:main']},
    long_description=read('README.md'),
    long_description_content_type='text/markdown'
)
