#!/usr/bin/env python
"""
The python hessian affine keypoint module

Command Line:
    python -m pyhesaff detect_feats --show --siftPower=0.5 --maxBinValue=-1
    python -m pyhesaff detect_feats --show
    python -m pyhesaff detect_feats --show --siftPower=0.5,
"""
import numpy as np
import ubelt as ub
from collections import OrderedDict
from pyhesaff import _hesaff

#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
vecs_dtype = np.uint8
img_dtype  = np.uint8
img32_dtype  = np.float32
# THE ORDER OF THIS LIST IS IMPORTANT!
HESAFF_TYPED_PARAMS = [
    # Pyramid Params
    (int,   'numberOfScales', 3),           # number of scale per octave
    (float, 'threshold', 16.0 / 3.0),       # noise dependent threshold on the response (sensitivity)
    (float, 'edgeEigenValueRatio', 10.0),   # ratio of the eigenvalues
    (int,   'border', 5),                   # number of pixels ignored at the border of image
    (int,   'maxPyramidLevels', -1),        # maximum number of pyramid divisions. -1 is no limit
    # Affine Shape Params
    (int,   'maxIterations', 16),           # number of affine shape interations
    (float, 'convergenceThreshold', 0.05),  # maximum deviation from isotropic shape at convergence
    (int,   'smmWindowSize', 19),           # width and height of the SMM (second moment matrix) mask
    (float, 'mrSize', 3.0 * np.sqrt(3.0)),  # size of the measurement region (as multiple of the feature scale)
    # SIFT params
    (int,   'spatialBins', 4),
    (int,   'orientationBins', 8),
    (float, 'maxBinValue', 0.2),
    # Shared params
    (float, 'initialSigma', 1.6),           # amount of smoothing applied to the initial level of first octave
    (int,   'patchSize', 41),               # width and height of the patch
    # My params
    (float, 'scale_min', -1.0),
    (float, 'scale_max', -1.0),
    (bool,  'rotation_invariance', False),
    (bool,  'augment_orientation', False),
    (float, 'ori_maxima_thresh', .8),
    (bool,  'affine_invariance', True),
    (bool,  'only_count', False),
    #
    (bool,  'use_dense', False),
    (int,   'dense_stride', 32),
    (float, 'siftPower', 1.0),
]

HESAFF_PARAM_DICT = OrderedDict([(key, val) for (type_, key, val) in HESAFF_TYPED_PARAMS])


def grab_test_imgpath(p='astro'):
    from pyhesaff._demodata import grab_test_image_fpath
    fpath = grab_test_image_fpath(p)
    # Old and broken
    # fpath = ub.grabdata('https://i.imgur.com/KXhKM72.png',
    #                     fname='astro.png',
    #                     hash_prefix='160b6e5989d2788c0296eac45b33e90fe612da23',
    #                     hasher='sha1')
    return fpath


def imread(fpath):
    import cv2
    return cv2.imread(fpath)


def _build_typed_params_kwargs_docstr_block(typed_params):
    r"""
    Args:
        typed_params (dict):

    CommandLine:
        python -m pyhesaff build_typed_params_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> typed_params = HESAFF_TYPED_PARAMS
        >>> result = build_typed_params_docstr(typed_params)
        >>> print(result)
    """
    kwargs_lines = []
    for tup in typed_params:
        type_, name, default = tup
        typestr = getattr(type_, '__name__', str(type_))
        line_fmtstr = '{name} ({typestr}): default={default}'
        line = line_fmtstr.format(name=name, typestr=typestr, default=default)
        kwargs_lines.append(line)
    kwargs_docstr_block = ('Kwargs:\n' + ub.indent('\n'.join(kwargs_lines), '    '))
    return ub.indent(kwargs_docstr_block, '    ')
hesaff_kwargs_docstr_block = _build_typed_params_kwargs_docstr_block(HESAFF_TYPED_PARAMS)


def argparse_hesaff_params():
    alias_dict = {'affine_invariance': 'ai'}
    alias_dict = {'rotation_invariance': 'ri'}
    default_dict_ = get_hesaff_default_params()
    try:
        import utool as ut
        hesskw = ut.argparse_dict(default_dict_, alias_dict=alias_dict)
    except Exception as ex:
        print('ex = {!r}'.format(ex))
        return default_dict_
    return hesskw


KPTS_DIM = _hesaff.get_kpts_dim()
DESC_DIM = _hesaff.get_desc_dim()


#============================
# helpers
#============================


def alloc_patches(nKpts, size=41):
    patches = np.empty((nKpts, size, size), np.float32)
    return patches


def alloc_vecs(nKpts):
    # array of bytes
    vecs = np.empty((nKpts, DESC_DIM), vecs_dtype)
    return vecs


def alloc_kpts(nKpts):
    # array of floats
    kpts = np.empty((nKpts, KPTS_DIM), kpts_dtype)
    #kpts = np.zeros((nKpts, KPTS_DIM), kpts_dtype) - 1.0  # array of floats
    return kpts


def _make_hesaff_cpp_params(kwargs):
    hesaff_params = HESAFF_PARAM_DICT.copy()
    for key, val in kwargs.items():
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print('[pyhesaff] WARNING: key=%r is not known' % key)
    return hesaff_params




#============================
# hesaff python interface
#============================


def get_hesaff_default_params():
    return HESAFF_PARAM_DICT.copy()


def get_is_debug_mode():
    return _hesaff.is_debug_mode()


def get_cpp_version():
    r"""
    Returns:
        int: cpp_version

    CommandLine:
        python -m pyhesaff get_cpp_version

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> cpp_version = get_cpp_version()
        >>> isdebug = get_is_debug_mode()
        >>> print('cpp_version = %r' % (cpp_version,))
        >>> print('isdebug = %r' % (isdebug,))
        >>> assert cpp_version == 4, 'cpp version mimatch'
    """

    #str_ptr = HESAFF_CLIB.cmake_build_type()
    # copy c string into python
    #pystr = ctypes.c_char_p(str_ptr).value
    # need to free c string
    #HESAFF_CLIB.free_char(str_ptr)
    #print('pystr = %r' % (pystr,))
    #print('pystr = %s' % (pystr,))
    return _hesaff.get_cpp_version()


# full detection and extraction


def detect_feats(img_fpath, use_adaptive_scale=False, nogravity_hack=False, **kwargs):
    r"""
    driver function for detecting hessian affine keypoints from an image path.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.

    Args:
        img_fpath (str): image file path on disk
        use_adaptive_scale (bool):
        nogravity_hack (bool):

    Kwargs:
        numberOfScales (int)         : default=3
        threshold (float)            : default=5.33333333333
        edgeEigenValueRatio (float)  : default=10.0
        border (int)                 : default=5
        maxIterations (int)          : default=16
        convergenceThreshold (float) : default=0.05
        smmWindowSize (int)          : default=19
        mrSize (float)               : default=5.19615242271
        spatialBins (int)            : default=4
        orientationBins (int)        : default=8
        maxBinValue (float)          : default=0.2
        initialSigma (float)         : default=1.6
        patchSize (int)              : default=41
        scale_min (float)            : default=-1.0
        scale_max (float)            : default=-1.0
        rotation_invariance (bool)   : default=False
        affine_invariance (bool)     : default=True

    Returns:
        tuple : (kpts, vecs)

    CommandLine:
        python -m pyhesaff detect_feats
        python -m pyhesaff detect_feats --show
        python -m pyhesaff detect_feats --show --fname star.png
        python -m pyhesaff detect_feats --show --fname zebra.png
        python -m pyhesaff detect_feats --show --fname astro.png
        python -m pyhesaff detect_feats --show --fname carl.jpg

        python -m pyhesaff detect_feats --show --fname astro.png --ri
        python -m pyhesaff detect_feats --show --fname astro.png --ai

        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --numberOfScales=1 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-max=100 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-min=20 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-min=100 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-max=20 --verbose

        python -m vtool.test_constrained_matching visualize_matches --show
        python -m vtool.tests.dummy testdata_ratio_matches --show

        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-min=35 --scale-max=40

        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-max=40 --darken .5
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-max=30 --darken .5
        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-max=30 --darken .5

        # DENSE KEYPOINTS
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64 --rotation-invariance
        python -m pyhesaff detect_feats --show --fname astro.png \
            --affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=3 \
            --maxPyramidLevels=2 --use_dense --dense_stride=32

        python -m pyhesaff detect_feats --show --only_count=False

    Example0:
        >>> # ENABLE_DOCTEST
        >>> # Test simple detect
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> TAU = 2 * np.pi
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro'))
        >>> kwargs = argparse_hesaff_params()
        >>> print('kwargs = %r' % (kwargs,))
        >>> (kpts, vecs) = detect_feats(img_fpath, **kwargs)
        >>> # Show keypoints
        >>> # xdoctest: +REQUIRES(--show)
        >>> imgBGR = imread(img_fpath)
        >>> # take a random stample
        >>> frac = ub.argval('--frac', default=1.0)
        >>> print('frac = %r' % (frac,))
        >>> idxs = vecs[0:int(len(vecs) * frac)]
        >>> vecs, kpts = vecs[idxs], kpts[idxs]
        >>> default_showkw = dict(ori=False, ell=True, ell_linewidth=2,
        >>>                       ell_alpha=.4, ell_color='distinct')
        >>> print('default_showkw = %r' % (default_showkw,))
        >>> #showkw = ut.argparse_dict(default_showkw)
        >>> #import plottool as pt
        >>> #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, **showkw)
        >>> #pt.show_if_requested()
    """
    # Load image
    kpts, vecs = _hesaff.detect_fpath(img_fpath, **kwargs)
    if use_adaptive_scale:  # Adapt scale if requested
        kpts, vecs = adapt_scale(img_fpath, kpts)
    if nogravity_hack:
        kpts, vecs = vtool_adapt_rotation(img_fpath, kpts)
    return kpts, vecs


def detect_feats2(img_or_fpath, **kwargs):
    """
    General way of detecting from either an fpath or ndarray

    Args:
        img_or_fpath (str or ndarray):  file path string

    Returns:
        tuple
    """
    if isinstance(img_or_fpath, str):
        fpath = img_or_fpath
        return detect_feats(fpath, **kwargs)
    else:
        img = img_or_fpath
        return detect_feats_in_image(img, **kwargs)


def detect_feats_list(image_paths_list, **kwargs):
    """
    Args:
        image_paths_list (list): A list of image paths

    Returns:
        tuple: (kpts_list, vecs_list) A tuple of lists of keypoints and decsriptors

    Kwargs:
        numberOfScales (int)         : default=3
        threshold (float)            : default=5.33333333333
        edgeEigenValueRatio (float)  : default=10.0
        border (int)                 : default=5
        maxIterations (int)          : default=16
        convergenceThreshold (float) : default=0.05
        smmWindowSize (int)          : default=19
        mrSize (float)               : default=5.19615242271
        spatialBins (int)            : default=4
        orientationBins (int)        : default=8
        maxBinValue (float)          : default=0.2
        initialSigma (float)         : default=1.6
        patchSize (int)              : default=41
        scale_min (float)            : default=-1.0
        scale_max (float)            : default=-1.0
        rotation_invariance (bool)   : default=False

    CommandLine:
        python -m pyhesaff._pyhesaff detect_feats_list --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> fpath = grab_test_imgpath('astro')
        >>> image_paths_list = [grab_test_imgpath('carl'), grab_test_imgpath('superstar'), fpath]
        >>> (kpts_list, vecs_list) = detect_feats_list(image_paths_list)
        >>> #print((kpts_list, vecs_list))
        >>> # Assert that the normal version agrees
        >>> serial_list = [detect_feats(fpath) for fpath in image_paths_list]
        >>> kpts_list2 = [c[0] for c in serial_list]
        >>> vecs_list2 = [c[1] for c in serial_list]
        >>> diff_kpts = [kpts - kpts2 for kpts, kpts2 in zip(kpts_list, kpts_list2)]
        >>> diff_vecs = [vecs - vecs2 for vecs, vecs2 in zip(vecs_list, vecs_list2)]
        >>> assert all([x.sum() == 0 for x in diff_kpts]), 'inconsistent results'
        >>> assert all([x.sum() == 0 for x in diff_vecs]), 'inconsistent results'
    """
    results = [detect_feats(path, **kwargs) for path in image_paths_list]
    kpts_list = [item[0] for item in results]
    vecs_list = [item[1] for item in results]
    return kpts_list, vecs_list


def detect_feats_in_image(img, **kwargs):
    r"""
    Takes a preloaded image and detects keypoints and descriptors

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data, should be in BGR or grayscale

    Returns:
        tuple: (kpts, vecs)

    CommandLine:
        python -m pyhesaff detect_feats_in_image --show
        python -m pyhesaff detect_feats_in_image --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('astro')
        >>> img = imread(img_fpath)
        >>> (kpts, vecs) = detect_feats_in_image(img)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.interact_keypoints.ishow_keypoints(img, kpts, vecs, ori=True,
        >>>                                       ell_alpha=.4, color='distinct')
        >>> pt.set_figtitle('Detect Kpts in Image')
        >>> pt.show_if_requested()
    """
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    return _hesaff.detect_image(img, **kwargs)


def detect_num_feats_in_image(img, **kwargs):
    r"""
    Just quickly returns how many keypoints are in the image. Does not attempt
    to return or store the values.

    It is a good idea to turn off things like ai and ri here.

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data

    Returns:
        int: nKpts

    ISSUE: there seems to be an inconsistency for jpgs between this and detect_feats

    CommandLine:
        python -m pyhesaff detect_num_feats_in_image:0 --show
        python -m pyhesaff detect_num_feats_in_image:1 --show
        python -m xdoctest pyhesaff detect_num_feats_in_image:0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('astro')
        >>> img = imread(img_fpath)
        >>> nKpts = detect_num_feats_in_image(img)
        >>> kpts, vecs = detect_feats_in_image(img)
        >>> #assert nKpts == len(kpts), 'inconsistency'
        >>> result = ('nKpts = %s' % (ub.repr2(nKpts),))
        >>> print(result)

    Example1:
        >>> # TIMEDOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> setup = ub.codeblock(
            '''
            import pyhesaff
            img_fpath = grab_test_imgpath('carl')
            img = imread(img_fpath)
            ''')
        >>> stmt_list = [
        >>>    'pyhesaff.detect_feats_in_image(img)',
        >>>    'pyhesaff.detect_num_feats_in_image(img, affine_invariance=False)',
        >>>    'pyhesaff.detect_num_feats_in_image(img)',
        >>> ]
        >>> iterations = 30
        >>> verbose = True
        >>> #ut.timeit_compare(stmt_list, setup=setup, iterations=iterations,
        >>> #                  verbose=verbose, assertsame=False)

    """
    # We dont need to find vectors at all here
    kwargs['only_count'] = True
    #kwargs['only_count'] = False
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    return _hesaff.count_image(img, **kwargs)


# just extraction


def extract_vecs(img_fpath, kpts, **kwargs):
    r"""
    Extract SIFT descriptors at keypoint locations

    Args:
        img_fpath (str):
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[uint8_t, ndim=2]: vecs -  descriptor vectors

    CommandLine:
        python -m pyhesaff extract_vecs:0
        python -m pyhesaff extract_vecs:1 --fname=astro.png
        python -m pyhesaff extract_vecs:1 --fname=patsy.jpg --show
        python -m pyhesaff extract_vecs:1 --fname=carl.jpg
        python -m pyhesaff extract_vecs:1 --fname=zebra.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('carl')
        >>> kpts = np.array([[20, 25, 5.21657705, -5.11095951, 24.1498699, 0],
        >>>                  [29, 25, 2.35508823, -5.11095952, 24.1498692, 0],
        >>>                  [30, 30, 12.2165705, 12.01909553, 10.5286992, 0],
        >>>                  [31, 29, 13.3555705, 17.63429554, 14.1040992, 0],
        >>>                  [32, 31, 16.0527005, 3.407312351, 11.7353722, 0]], dtype=np.float32)
        >>> vecs = extract_vecs(img_fpath, kpts)
        >>> result = 'vecs = {}'.format(vecs)
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('astro')
        >>> # Extract original keypoints
        >>> kpts, vecs1 = detect_feats(img_fpath)
        >>> # Re-extract keypoints
        >>> vecs2 = extract_vecs(img_fpath, kpts)
        >>> # Descriptors should be the same
        >>> errors = (vecs1.astype(float) - vecs2.astype(float)).sum(axis=1)
        >>> errors_index = np.nonzero(errors)[0]
        >>> print('errors = %r' % (errors,))
        >>> print('errors_index = %r' % (errors_index,))
        >>> print('errors.sum() = %r' % (errors.sum(),))
        >>> # VISUALIZTION
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> # Extract the underlying grayscale patches
        >>> img = imread(img_fpath)
        >>> patch_list = extract_patches(img, kpts)
        >>> pt.interact_keypoints.ishow_keypoints(img_fpath, kpts[errors_index], vecs1[errors_index], fnum=1)
        >>> ax = pt.draw_patches_and_sifts(patch_list[errors_index], vecs1[errors_index], pnum=(1, 2, 1), fnum=2)
        >>> ax.set_title('patch extracted')
        >>> ax = pt.draw_patches_and_sifts(patch_list[errors_index], vecs2[errors_index], pnum=(1, 2, 2), fnum=2)
        >>> ax.set_title('image extracted')
        >>> pt.set_figtitle('Error Keypoints')
        >>> pt.show_if_requested()
    """
    kpts = np.ascontiguousarray(kpts, dtype=kpts_dtype)
    if isinstance(img_fpath, str):
        return _hesaff.extract_desc_fpath(img_fpath, kpts, **kwargs)
    return _hesaff.extract_desc_image(img_fpath, kpts, **kwargs)


def extract_patches(img_or_fpath, kpts, **kwargs):
    r"""
    Extract patches used to compute SIFT descriptors.

    Args:
        img_or_fpath (ndarray or str):
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m pyhesaff extract_patches:0 --show
        python -m pyhesaff extract_vecs:1 --fname=astro.png
        python -m pyhesaff extract_vecs:1 --fname=patsy.jpg --show
        python -m pyhesaff extract_vecs:1 --fname=carl.jpg
        python -m pyhesaff extract_vecs:1 --fname=zebra.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:vtool)
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> kwargs = {}
        >>> img_fpath = grab_test_imgpath('carl')
        >>> img = imread(img_fpath)
        >>> img_or_fpath = img
        >>> kpts, vecs1 = detect_feats(img_fpath)
        >>> kpts = kpts[1::len(kpts) // 9]
        >>> vecs1 = vecs1[1::len(vecs1) // 9]
        >>> cpp_patch_list = extract_patches(img, kpts)
        >>> py_patch_list_ = np.array(vt.get_warped_patches(img_or_fpath, kpts, patch_size=41)[0])
        >>> py_patch_list = np.array(vt.convert_image_list_colorspace(py_patch_list_, 'gray'))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> ax = pt.draw_patches_and_sifts(cpp_patch_list, None, pnum=(1, 2, 1))
        >>> ax.set_title('C++ extracted')
        >>> ax = pt.draw_patches_and_sifts(py_patch_list, None, pnum=(1, 2, 2))
        >>> ax.set_title('Python extracted')
        >>> pt.show_if_requested()
    """
    kpts = np.ascontiguousarray(kpts, dtype=kpts_dtype)
    if isinstance(img_or_fpath, str):
        return _hesaff.extract_patches_fpath(img_or_fpath, kpts, **kwargs)
    return _hesaff.extract_patches_image(img_or_fpath, kpts, **kwargs)


def extract_desc_from_patches(patch_list):
    r"""
    Careful about the way the patches are extracted here.

    Args:
        patch_list (ndarray[ndims=3]):

    CommandLine:
        python -m pyhesaff extract_desc_from_patches
        python -m pyhesaff extract_desc_from_patches  --show
        python -m pyhesaff extract_desc_from_patches:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:vtool)
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro'))
        >>> # First extract keypoints normally
        >>> (orig_kpts_list, orig_vecs_list) = detect_feats(img_fpath)
        >>> # Take 9 keypoints
        >>> img = imread(img_fpath)
        >>> kpts_list = orig_kpts_list[1::len(orig_kpts_list) // 9]
        >>> vecs_list = orig_vecs_list[1::len(orig_vecs_list) // 9]
        >>> # Extract the underlying grayscale patches (using different patch_size)
        >>> patch_list_ = np.array(vt.get_warped_patches(img, kpts_list, patch_size=64)[0])
        >>> patch_list = np.array(vt.convert_image_list_colorspace(patch_list_, 'gray'))
        >>> # Extract descriptors from the patches
        >>> vecs_array = extract_desc_from_patches(patch_list)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro'))
        >>> # First extract keypoints normally
        >>> (orig_kpts_list, orig_vecs_list) = detect_feats(img_fpath)
        >>> # Take 9 keypoints
        >>> img = imread(img_fpath)
        >>> kpts_list = orig_kpts_list[1::len(orig_kpts_list) // 9]
        >>> vecs_list = orig_vecs_list[1::len(orig_vecs_list) // 9]
        >>> # Extract the underlying grayscale patches
        >>> patch_list = extract_patches(img, kpts_list)
        >>> patch_list = np.round(patch_list).astype(np.uint8)
        >>> # Currently its impossible to get the correct answer
        >>> # TODO: allow patches to be passed in as float32
        >>> # Extract descriptors from those patches
        >>> vecs_array = extract_desc_from_patches(patch_list)
        >>> # Comparse to see if they are close to the original descriptors
        >>> errors = (vecs_list.astype(float) - vecs_array.astype(float)).sum(axis=1)
        >>> print('Errors: %r' % (errors,))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> ax = pt.draw_patches_and_sifts(patch_list, vecs_array, pnum=(1, 2, 1))
        >>> ax.set_title('patch extracted')
        >>> ax = pt.draw_patches_and_sifts(patch_list, vecs_list, pnum=(1, 2, 2))
        >>> ax.set_title('image extracted')
        >>> pt.show_if_requested()
    """
    ndims = len(patch_list.shape)
    if ndims == 4 and patch_list.shape[-1] == 1:
        print('[pyhesaff] warning need to reshape patch_list')
        # need to remove grayscale dimension, maybe it should be included
        patch_list = patch_list.reshape(patch_list.shape[0:3])
    elif ndims == 4 and patch_list.shape[-1] == 3:
        assert False, 'cannot handle color images yet'
    assert patch_list.flags['C_CONTIGUOUS'], 'patch_list must be contiguous array'
    # If the input array list is memmaped it is a good idea to process in chunks
    CHUNKS = isinstance(patch_list, np.memmap)
    if not CHUNKS:
        return _hesaff.extract_desc_from_patches(patch_list)
    chunksize = 2048
    num_patches = patch_list.shape[0]
    vecs_array = alloc_vecs(num_patches)
    _iter = range(num_patches // chunksize)
    _progiter = ub.ProgIter(_iter, desc='extracting sift chunk')
    for ix in _progiter:
        lx = ix * chunksize
        rx = (ix + 1) * chunksize
        patch_sublist = np.array(patch_list[lx:rx])
        vecs_array[lx:rx] = _hesaff.extract_desc_from_patches(patch_sublist)
    last_size = num_patches - rx
    if last_size > 0:
        lx = rx
        rx = lx + last_size
        patch_sublist = np.array(patch_list[lx:rx])
        vecs_array[lx:rx] = _hesaff.extract_desc_from_patches(patch_sublist)
    return vecs_array

#============================
# other
#============================


def test_rot_invar():
    r"""
    CommandLine:
        python -m pyhesaff test_rot_invar --show
        python -m pyhesaff test_rot_invar --show --nocpp

        python -m vtool.tests.dummy testdata_ratio_matches --show --ratio_thresh=1.0 --rotation_invariance
        python -m vtool.tests.dummy testdata_ratio_matches --show --ratio_thresh=1.1 --rotation_invariance

    Example:
        >>> # DISABLE_DODCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> test_rot_invar()
    """
    import cv2
    import vtool as vt
    import plottool_ibeis as pt
    TAU = 2 * np.pi
    fnum = pt.next_fnum()
    NUM_PTS = 5  # 9
    theta_list = np.linspace(0, TAU, NUM_PTS, endpoint=False)
    nRows, nCols = pt.get_square_row_cols(len(theta_list), fix=True)
    next_pnum = pt.make_pnum_nextgen(nRows, nCols)
    # Expand the border a bit around star.png
    pad_ = 100
    img_fpath = grab_test_imgpath('superstar')
    img_fpath2 = vt.pad_image_ondisk(img_fpath, pad_, value=26)
    for theta in theta_list:
        print('-----------------')
        print('theta = %r' % (theta,))
        img_fpath = vt.rotate_image_ondisk(img_fpath2, theta, border_mode=cv2.BORDER_REPLICATE)
        if not ub.argflag('--nocpp'):
            (kpts_list_ri, vecs_list2) = detect_feats(img_fpath, rotation_invariance=True)
            kpts_ri = kpts_list_ri[0:2]
        (kpts_list_gv, vecs_list1) = detect_feats(img_fpath, rotation_invariance=False)
        kpts_gv = kpts_list_gv[0:2]
        # find_kpts_direction
        imgBGR = imread(img_fpath)
        kpts_ripy = vt.find_kpts_direction(imgBGR, kpts_gv, DEBUG_ROTINVAR=False)
        # Verify results stdout
        #print('nkpts = %r' % (len(kpts_gv)))
        #print(vt.kpts_repr(kpts_gv))
        #print(vt.kpts_repr(kpts_ri))
        #print(vt.kpts_repr(kpts_ripy))
        # Verify results plot
        pt.figure(fnum=fnum, pnum=next_pnum())
        pt.imshow(imgBGR)
        #if len(kpts_gv) > 0:
        #    pt.draw_kpts2(kpts_gv, ori=True, ell_color=pt.BLUE, ell_linewidth=10.5)
        ell = False
        rect = True
        if not ub.argflag('--nocpp'):
            if len(kpts_ri) > 0:
                pt.draw_kpts2(kpts_ri, rect=rect, ell=ell, ori=True,
                              ell_color=pt.RED, ell_linewidth=5.5)
        if len(kpts_ripy) > 0:
            pt.draw_kpts2(kpts_ripy, rect=rect, ell=ell,  ori=True,
                          ell_color=pt.GREEN, ell_linewidth=3.5)
    pt.set_figtitle('green=python, red=C++')
    pt.show_if_requested()


def vtool_adapt_rotation(img_fpath, kpts):
    """ rotation invariance in python """
    import vtool.patch as ptool
    import vtool.image as gtool
    imgBGR = gtool.imread(img_fpath)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


def adapt_scale(img_fpath, kpts):
    import vtool.ellipse as etool
    nScales = 16
    nSamples = 16
    low, high = -1, 2
    kpts2 = etool.adaptive_scale(img_fpath, kpts, nScales, low, high, nSamples)
    # passing in 0 orientation results in gravity vector direction keypoint
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


# del type_, key, val

if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff._pyhesaff
        python -m pyhesaff._pyhesaff --allexamples
        python -m pyhesaff._pyhesaff --allexamples --noface --nosrc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
