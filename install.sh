#!/bin/bash

CWD=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit

echo "Creating .whl file"
python setup.py bdist_wheel
echo "Installing package"
pip install "./dist/py_lib-0.1.0-py3-none-any.whl"
echo "Package installed!"

cd "$CWD" || exit