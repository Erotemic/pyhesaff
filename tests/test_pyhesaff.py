import cv2


def interactive_test_pyheaff(img_fpath):
    r"""
    This show is interactive in this test!

    Args:
        img_fpath (str):
    """
    import pytest
    pytest.skip('Broken in CI')

    import pyhesaff
    import ubelt as ub
    kpts, desc = pyhesaff.detect_feats(img_fpath)
    rchip = cv2.imread(img_fpath)
    if ub.argflag('--show'):
        from plottool.interact_keypoints import ishow_keypoints
        ishow_keypoints(rchip, kpts, desc)
    return locals()


if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
