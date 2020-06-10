#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import sys
from os.path import exists
from collections import OrderedDict


def native_mb_python_tag(plat_impl=None, version_info=None):
    """
    Example:
        >>> print(native_mb_python_tag())
        >>> print(native_mb_python_tag('PyPy', (2, 7)))
        >>> print(native_mb_python_tag('CPython', (3, 8)))
    """
    if plat_impl is None:
        import platform
        plat_impl = platform.python_implementation()

    if version_info is None:
        import sys
        version_info = sys.version_info

    major, minor = version_info[0:2]
    ver = '{}{}'.format(major, minor)

    if plat_impl == 'CPython':
        # TODO: get if cp27m or cp27mu
        impl = 'cp'
        if ver == '27':
            IS_27_BUILT_WITH_UNICODE = True  # how to determine this?
            if IS_27_BUILT_WITH_UNICODE:
                abi = 'mu'
            else:
                abi = 'm'
        else:
            if ver == '38':
                # no abi in 38?
                abi = ''
            else:
                abi = 'm'
        mb_tag = '{impl}{ver}-{impl}{ver}{abi}'.format(**locals())
    elif plat_impl == 'PyPy':
        abi = ''
        impl = 'pypy'
        ver = '{}{}'.format(major, minor)
        mb_tag = '{impl}-{ver}'.format(**locals())
    else:
        raise NotImplementedError(plat_impl)
    return mb_tag


def parse_version(fpath='pyhesaff/__init__.py'):
    """
    Statically parse the version number from a python file


    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_long_description(fpath='README.rst'):
    """
    Reads README text, but doesn't break if README does not exist.
    """
    if exists(fpath):
        with open(fpath, 'r') as file:
            return file.read()
    return ''


def parse_requirements(fname='requirements.txt'):
    """
    Parse the package dependencies listed in a requirements file but
    strips specific versioning information.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        elif line.startswith('-e '):
            info = {}
            info['package'] = line.split('#egg=')[1]
            yield info
        else:
            # Remove versioning from the package
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]

            info = {}
            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    # Handle platform specific dependencies
                    # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest  # NOQA
                info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    # This breaks on pip install, so check that it exists.
    packages = []
    if exists(require_fpath):
        for info in parse_require_file(require_fpath):
            package = info['package']
            if not sys.version.startswith('3.4'):
                # apparently package_deps are broken in 3.4
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    package += ';' + platform_deps
            packages.append(package)
    return packages


try:
    class EmptyListWithLength(list):
        def __len__(self):
            return 1

        def __repr__(self):
            return 'EmptyListWithLength()'

        def __str__(self):
            return 'EmptyListWithLength()'
except Exception:
    raise RuntimeError('FAILED TO ADD BUILD CONSTRUCTS')


NAME = 'wbia-pyhesaff'


MB_PYTHON_TAG = native_mb_python_tag()  # NOQA
VERSION = version = parse_version('pyhesaff/__init__.py')  # must be global for git tags

ORIGINAL_AUTHORS = ['Krystian Mikolajczyk', 'Michal Perdoch']
EXTENDED_AUTHORS = ['Jon Crall', 'Avi Weinstock']
CURRENT_AUTHORS = ['WildMe Developers']

AUTHORS = ', '.join(ORIGINAL_AUTHORS + EXTENDED_AUTHORS + CURRENT_AUTHORS)
AUTHOR_EMAIL = 'dev@wildme.org'
URL = 'https://github.com/WildbookOrg/hesaff'
LICENSE = 'APL 2.0'
DESCRIPTION = 'PyHesaff - Python wrapper for Hessian Affine'


KWARGS = OrderedDict(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=parse_long_description('README.rst'),
    long_description_content_type='text/x-rst',
    url=URL,
    license=LICENSE,
    install_requires=parse_requirements('requirements.txt'),
    extras_require={},
    # --- PACKAGES ---
    # The combination of packages and package_dir is how scikit-build will
    # know that the cmake installed files belong in the pyhesaff module and
    # not the data directory.
    packages=[
        'pyhesaff',
        # These are generated modules that will be created via build
        'pyhesaff.lib',
    ],
    package_dir={
        'pyhesaff': 'pyhesaff',
        # Note: this requires that HESAFF_LIB_INSTALL_DIR is set to pyhesaff/lib
        # in the src/cpp/CMakeLists.txt
        'pyhesaff.lib': 'pyhesaff/lib',
    },
    include_package_data=False,
    # List of classifiers available at:
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 6 - Mature',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    cmake_args=[
        '-DBUILD_TESTS=OFF',
        '-DBUILD_DOC=OFF',
    ],
    ext_modules=EmptyListWithLength(),  # hack for including ctypes bins
)

if __name__ == '__main__':
    """
    python -c "import pyhesaff; print(pyhesaff.__file__)"
    """
    import skbuild
    skbuild.setup(**KWARGS)
