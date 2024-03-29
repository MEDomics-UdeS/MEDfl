# MEDfl: Federated Learning and Differential Privacy Simulation Tool for Tabular Data
![Python Versions](https://img.shields.io/badge/python-3.9-blue)
![Build Status](https://travis-ci.org/MEDomics-UdeS/MEDfl.svg?branch=main)

![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)
![License: MIT](https://img.shields.io/badge/license-MIT-green)



## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Documentation](#3-documentation)
  * [4. Getting started](#4-Getting-started)
  * [5. Acknowledgement](#5-acknowledgement)
  * [6. Authors](#6-authors)

## 1. Introduction
This Python package is an open-source tool designed for simulating federated learning and incorporating differential privacy. It empowers researchers and developers to effortlessly create, execute, and assess federated learning pipelines while seamlessly working with various tabular datasets.


## 2. Installation

### Python installation
The MEDfl package requires *python 3.9* or more to be run. If you don't have it installed  on your machine, check out the following link [Python](https://www.python.org/downloads/).
It also requires MySQL database.

### Package installation
For now, you can  install the ``MEDfl``package as:
```
git clone https://github.com/MEDomics-UdeS/MEDfl.git
cd MEDfl
pip install -e .
```
### MySQL DB configuration
MEDfl requires a MySQL DB connection, and this is in order to allow users to work with their own tabular datasets, we have created a bash script to install and configure A MySQL DB with phpmyadmin monitoring system, run the following command then change your credential on the MEDfl/scripts/base.py file
```
sudo bash MEDfl/scripts/setup_mysql.sh
```

## 3. Documentation
We used sphinx to create the documentation for this project.  you can generate and host it locally by compiling the documentation source code using:
```
cd docs
make clean
make html
```

Then open it locally using:

```
cd _build/html
python -m http.server
```

## 4. Getting started
We have created a complete tutorial for the different functionalities of the package. It can be found here: [Tutorial](https://github.com/MEDomics-UdeS/MEDfl/blob/main/notebooks/First_Tuto.ipynb).


## 5. Acknowledgment
MEDfl is an open-source package that welcomes any contribution and feedback. We wish that this package could serve the growing research community in federated learning for health.

## 6. Authors
* [MEDomics-UdeS](https://www.medomics-udes.org/en/)
* [Hithem Lamri](https://github.com/HaithemLamri)

