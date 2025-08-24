"""
This helps find the shared library that contains the compiled subroutines.
Its a bit hacky and could use a cleanup by someone who really understands
how python c-extension libraries are named and placed depending on system.
"""
from os.path import join, exists, normpath
import sys
import os
import ctypes


#============================
# general ctypes interface
#============================

__DEBUG_CLIB__ = '--debug' in sys.argv or '--debug-clib' in sys.argv


def get_plat_specifier():
    """
    Standard platform specifier used by distutils
    """
    try:
        import distutils
    except ImportError:
        return get_plat_specifier2()
    try:
        plat_name = distutils.util.get_platform()
    except AttributeError:
        plat_name = distutils.sys.platform
    plat_specifier = '.{}-{}'.format(plat_name, sys.version[0:3])
    if hasattr(sys, 'gettotalrefcount'):
        plat_specifier += '-pydebug'
    return plat_specifier


def _py_ver_str():
    """Return 'MAJOR.MINOR' (e.g., '3.12')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _norm_arch():
    """Normalize common architecture names to the ones your code expects."""
    import platform
    m = (platform.machine() or "").lower()
    if m in {"x86_64", "amd64"}:
        return "x86_64"
    if m in {"i386", "i686", "x86"}:
        return "i686"
    if m in {"aarch64", "arm64"}:
        return "arm64"
    return m or ("x86_64" if sys.maxsize > 2**32 else "i686")


def get_plat_specifier2():
    """
    Standard platform specifier (distutils-free).
    Mirrors your existing format: '.<plat>-<pyver>' + optional '-pydebug'.
    """
    import sysconfig
    plat_name = sysconfig.get_platform() or sys.platform
    plat_specifier = f".{plat_name}-{_py_ver_str()}"
    if hasattr(sys, "gettotalrefcount"):  # CPython debug builds
        plat_specifier += "-pydebug"
    return plat_specifier


def get_candidate_plat_specifiers2():
    """
    Produce a list of plausible platform suffixes without using distutils.
    Keeps your legacy candidates and adds a few modern ones (manylinux, macOS).
    """
    import sysconfig
    arch = _norm_arch()
    py_ver = _py_ver_str()
    plat_name = sysconfig.get_platform() or sys.platform

    plat_name_cands = [plat_name]

    if sys.platform.startswith("linux"):
        # Keep broad fallbacks and add some manylinux variants that show up in practice.
        plat_name_cands += [
            "linux",
            "manylinux",
            "manylinux1",
            "manylinux2010",
            "manylinux2014",
        ]
        # Wheel-style tags sometimes include glibc floor; include a couple likely ones.
        # (Your filenames use '-' not '_', but we’ll keep your format below.)
        if arch:
            plat_name_cands += [
                f"manylinux_2_17_{arch}",
                f"manylinux_2_5_{arch}",
            ]

    elif sys.platform.startswith("darwin"):
        # Keep your historical macOS entries; add modern versions and universal2.
        plat_name_cands += [
            "macosx-10.6",
            "macosx-10.7",
            "macosx-10.9",
            "macosx-10.12",
            "macosx-11.0",
            "macosx-12.0",
            "macosx-13.0",
            "macosx-10.6-intel",
            "macosx-10.7-intel",
            "macosx-10.9-intel",
            "macosx-10.12-intel",
            "macosx-11.0-universal2",
            "macosx-12.0-universal2",
        ]

    elif sys.platform.startswith("win32"):
        # Keep both in case filenames vary.
        plat_name_cands += ["win-amd64", "win32"]

    spec_list = []
    for pn in plat_name_cands:
        spec_list.extend([
            f".{pn}-{py_ver}",
            f".{pn}-{arch}-{py_ver}",
        ])

    # Bare suffix (your original behavior)
    spec_list.append("")
    return spec_list


def get_candidate_plat_specifiers():
    try:
        import distutils
    except ImportError:
        return get_candidate_plat_specifiers2()
    if sys.maxsize > 2 ** 32:
        arch = 'x86_64'  # TODO: get correct arch spec
    else:
        arch = 'i686'  # TODO: get correct arch spec

    py_ver = sys.version[0:3]

    try:
        plat_name = distutils.util.get_platform()
    except AttributeError:
        plat_name = distutils.sys.platform

    plat_name_cands = [plat_name]
    if sys.platform.startswith('linux'):
        plat_name_cands.append('linux')
        plat_name_cands.append('manylinux1')
        plat_name_cands.append('manylinux')
    elif sys.platform.startswith('darwin'):
        # HACK:
        # on travis, wheel builds as libhesaff.macosx-10.12-x86_64-2.7.dylib,
        # but we seem to want libhesaff.macosx-10.6-intel-2.7.dylib
        # TODO: what is the proper way to determine the ABI tag?
        plat_name_cands.append('macosx-10.6')
        plat_name_cands.append('macosx-10.7')
        plat_name_cands.append('macosx-10.9')
        plat_name_cands.append('macosx-10.12')
        plat_name_cands.append('macosx-10.6-intel')
        plat_name_cands.append('macosx-10.7-intel')
        plat_name_cands.append('macosx-10.9-intel')
        plat_name_cands.append('macosx-10.12-intel')
    elif sys.platform.startswith('win32'):
        # hack for win32
        plat_name_cands.append('win-amd64')
        pass

    spec_list = []
    for plat_name in plat_name_cands:
        spec_list.extend([
            '.{}-{}'.format(plat_name, sys.version[0:3]),
            '.{}-{}-{}'.format(plat_name, arch, py_ver),
        ])
    spec_list.append('')
    return spec_list


def get_lib_fname_candidates(libname):
    """
    Args:
        libname (str): library name (e.g. 'hesaff', not 'libhesaff')

    Returns:
        list: libnames - list of plausible library file names

    CommandLine:
        python -m pyhesaff.ctypes_interface get_lib_fname_candidates

    Example:
        >>> from pyhesaff.ctypes_interface import *  # NOQA
        >>> libname = 'hesaff'
        >>> libnames = get_lib_fname_candidates(libname)
        >>> import ubelt as ub
        >>> print('libnames = {}'.format(ub.repr2(libnames)))
    """
    spec_list = get_candidate_plat_specifiers()

    prefix_list = ['lib' + libname]
    if sys.platform.startswith('win32'):
        # windows doesnt start names with lib
        prefix_list.append(libname)
        ext = '.dll'
    elif sys.platform.startswith('darwin'):
        ext = '.dylib'
    elif sys.platform.startswith('linux'):
        ext = '.so'
    else:
        raise Exception('Unknown operating system: %s' % sys.platform)
    # Construct priority ordering of libnames
    libnames = [''.join((prefix, spec, ext))
                for spec in spec_list
                for prefix in prefix_list]
    return libnames


def get_lib_dpath_list(root_dir):
    """
    input <root_dir>: deepest directory to look for a library (dll, so, dylib)
    returns <libnames>: list of plausible directories to look.
    """
    'returns possible lib locations'
    get_lib_dpath_list = [
        root_dir,
        # join(root_dir, 'lib'),
        # join(root_dir, 'build'),
        # join(root_dir, 'build', 'lib'),
    ]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, verbose=False):
    """ Search for the library """
    lib_fname_list = get_lib_fname_candidates(libname)
    tried_fpaths = []

    FINAL_LIB_FPATH = None

    for lib_fname in lib_fname_list:
        if verbose:
            print('--')
        curr_dpath = root_dir
        # max_depth = 0

        for lib_dpath in get_lib_dpath_list(curr_dpath):
            lib_fpath = normpath(join(lib_dpath, lib_fname))
            tried_fpaths.append(lib_fpath)
            flag = exists(lib_fpath)
            if verbose:
                print('[c] Check: {}, exists={}'.format(lib_fpath, int(flag)))
            if flag:
                if verbose:
                    print('using: {}'.format(lib_fpath))
                FINAL_LIB_FPATH = lib_fpath
                return lib_fpath

    if FINAL_LIB_FPATH is not None:
        return FINAL_LIB_FPATH
    else:
        contents = os.listdir(root_dir)
        msg = ('\n[C!] find_lib_fpath(libname={!r}, root_dir={!r})'.format(
               libname, root_dir) +
               '\n[c!] Cannot FIND dynamic library')
        print(msg)
        print('\n[c!] Checked: '.join(tried_fpaths))
        print('UNABLE TO FIND LIB IN DPATH contents = {!r}'.format(contents))
        raise ImportError(msg)


def load_clib(libname, root_dir):
    """
    Searches for a library matching libname and loads it

    Args:
        libname:  library name (e.g. 'hesaff', not 'libhesaff')

        root_dir: the directory that should contain the
                  library file (dll, dylib, or so).
    Returns:
        clib: a ctypes object used to interface with the library
    """
    ex = None
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        if sys.platform.startswith('win32'):
            clib = ctypes.windll[lib_fpath]
        else:
            clib = ctypes.cdll[lib_fpath]
    except OSError as ex_:
        ex = ex_
        print('[C!] Caught OSError:\n{!r}'.format(ex))
        errsuffix = 'Is there a missing dependency?'
    except Exception as ex_:
        ex = ex_
        print('[C!] Caught Exception:\n{!r}'.format(ex))
        errsuffix = 'Was the library correctly compiled?'
    else:
        def def_cfunc(return_type, func_name, arg_type_list):
            'Function to define the types that python needs to talk to c'
            cfunc = getattr(clib, func_name)
            cfunc.restype = return_type
            cfunc.argtypes = arg_type_list
        clib.__LIB_FPATH__ = lib_fpath
        return clib, def_cfunc, lib_fpath
    print('[C!] cwd={!r}'.format(os.getcwd()))
    print('[C!] load_clib(libname={!r}, root_dir={!r})'.format(libname, root_dir))
    print('[C!] lib_fpath = {!r}'.format(lib_fpath))
    errmsg = '[C] Cannot LOAD {!r} dynamic library. Caused by ex={!r}. {}'.format(libname, ex, errsuffix)
    print(errmsg)
    raise ImportError(errmsg)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pyhesaff.ctypes_interface
        python -m pyhesaff.ctypes_interface --allexamples
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
