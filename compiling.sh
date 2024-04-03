#!/bin/sh

python setup.py build_ext --inplace
python3 setup.py sdist bdist_wheel
