import numpy as np


def test_pybind_extension_smoke():
    import pyhesaff

    assert isinstance(pyhesaff.get_cpp_version(), int)
    img = np.zeros((32, 32), dtype=np.uint8)
    kpts, vecs = pyhesaff.detect_feats_in_image(img)
    assert kpts.dtype == np.float32
    assert vecs.dtype == np.uint8
    assert kpts.shape[1] == pyhesaff.KPTS_DIM
    assert vecs.shape[1] == pyhesaff.DESC_DIM
