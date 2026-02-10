#!/usr/bin/env python


def test_simple_iterative():
    import pytest

    pytest.skip('Broken in CI')

    import pyhesaff
    from pyhesaff._pyhesaff import grab_test_imgpath

    fpath_list = [
        grab_test_imgpath(),
    ]
    kpts_list = []

    for img_fpath in fpath_list:
        kpts, vecs = pyhesaff.detect_feats(img_fpath)
        print('img_fpath=%r' % img_fpath)
        print(f'kpts.shape={kpts.shape}')
        print(f'vecs.shape={vecs.shape}')
        assert len(kpts) == len(vecs)
        assert len(kpts) > 0, 'no keypoints were detected!'
        kpts_list.append(kpts)

    if 0:
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        fig = plt.figure()
        for i, fpath, kpts in enumerate(zip(fpath_list, kpts_list), start=1):
            ax = fig.add_subplot(2, 2, i)
            img = mpl.image.imread(fpath)
            plt.imshow(img)
            _xs, _ys = kpts.T[0:2]
            ax.plot(_xs, _ys, 'ro', alpha=0.5)


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
