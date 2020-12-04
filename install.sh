#!/bin/bash

CWD=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit

rm "./dist/*.whl" 2> /dev/null

echo "Creating .whl file"
python setup.py bdist_wheel
echo "Installing package"
pip install "./dist/*.whl"
echo "Package installed!"

cd "$CWD" || exit