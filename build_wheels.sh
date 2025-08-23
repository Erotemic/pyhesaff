#!/bin/bash
__doc__="""
SeeAlso:
    pyproject.toml
"""

LOCAL_CP_VERSION=$(python3 -c "import sys; print('cp' + ''.join(list(map(str, sys.version_info[0:2]))))")
echo "LOCAL_CP_VERSION = $LOCAL_CP_VERSION"

# Build for only the current version of Python
export CIBW_BUILD="${LOCAL_CP_VERSION}-*"
cibuildwheel --config-file pyproject.toml --platform linux --archs x86_64
