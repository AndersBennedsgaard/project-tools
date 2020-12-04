from setuptools import find_packages, setup


setup(
    name='py_lib', packages=find_packages(include='py_lib'),
    version='0.1.0', description='My library', author='Anders',
    license='MIT', install_requires=[]
)
