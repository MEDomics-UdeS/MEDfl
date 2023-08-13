import sys

from setuptools import find_packages, setup

# Check if current python installation is >= 3.8
if sys.version_info < (3, 8, 0):
  raise Exception("MEDflrequires python 3.8 or later")

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="MEDfl",
    version="0.1.0",
    author="MEDomics consortium",
    author_email="medomics.info@gmail.com",
    description="Python Open-source package for simulating federated learning and differential privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HaithemLamri/MEDfl",
    project_urls={
        'Documentation': 'https://',
        'Github': 'https://github.com/HaithemLamri/MEDfl'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Federated Learning',
        'Topic :: Scientific/Engineering ::Differential Privacy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='federated learning differential privacy medical research ',
    scripts=['scripts/setup_mysql.sh'],
    python_requires='>=3.8,<3.10',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements
)