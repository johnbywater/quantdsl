#!/usr/bin/env python
import os
import subprocess
import sys


def build_and_release(cwd):
    # Build and upload to PyPI.
    subprocess.check_call(['pip', 'install', '-U', 'pip', 'setuptools', 'twine'], cwd=cwd)
    subprocess.check_call(['rm', '-rf', 'dist/'], cwd=cwd)
    subprocess.check_call([sys.executable, 'setup.py', 'sdist'], cwd=cwd)
    subprocess.check_call(['twine', 'upload', 'dist/*'], cwd=cwd)


if __name__ == '__main__':
    cwd = os.path.join(os.environ['HOME'], 'PyCharmProjects', 'quantdsl')
    build_and_release(cwd)
