#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <string>

#include "hesaff_capi.h"

namespace py = pybind11;

struct HesaffParamsPy {
    int numberOfScales = 3;
    float threshold = 16.0f / 3.0f;
    float edgeEigenValueRatio = 10.0f;
    int border = 5;
    int maxPyramidLevels = -1;
    int maxIterations = 16;
    float convergenceThreshold = 0.05f;
    int smmWindowSize = 19;
    float mrSize = 3.0f * static_cast<float>(std::sqrt(3.0));
    int spatialBins = 4;
    int orientationBins = 8;
    float maxBinValue = 0.2f;
    float initialSigma = 1.6f;
    int patchSize = 41;
    float scale_min = -1.0f;
    float scale_max = -1.0f;
    bool rotation_invariance = false;
    bool augment_orientation = false;
    float ori_maxima_thresh = 0.8f;
    bool affine_invariance = true;
    bool only_count = false;
    bool use_dense = false;
    int dense_stride = 32;
    float siftPower = 1.0f;
};

struct DetectorHandle {
    explicit DetectorHandle(AffineHessianDetector* detector) : detector_(detector) {}
    DetectorHandle(const DetectorHandle&) = delete;
    DetectorHandle& operator=(const DetectorHandle&) = delete;
    ~DetectorHandle() {
        if (detector_) {
            free_hesaff(detector_);
        }
    }

    AffineHessianDetector* get() const { return detector_; }

private:
    AffineHessianDetector* detector_ = nullptr;
};

static void update_params_from_kwargs(HesaffParamsPy& params, const py::kwargs& kwargs) {
    for (const auto& item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        const py::handle value = item.second;
        if (key == "numberOfScales") {
            params.numberOfScales = py::cast<int>(value);
        } else if (key == "threshold") {
            params.threshold = py::cast<float>(value);
        } else if (key == "edgeEigenValueRatio") {
            params.edgeEigenValueRatio = py::cast<float>(value);
        } else if (key == "border") {
            params.border = py::cast<int>(value);
        } else if (key == "maxPyramidLevels") {
            params.maxPyramidLevels = py::cast<int>(value);
        } else if (key == "maxIterations") {
            params.maxIterations = py::cast<int>(value);
        } else if (key == "convergenceThreshold") {
            params.convergenceThreshold = py::cast<float>(value);
        } else if (key == "smmWindowSize") {
            params.smmWindowSize = py::cast<int>(value);
        } else if (key == "mrSize") {
            params.mrSize = py::cast<float>(value);
        } else if (key == "spatialBins") {
            params.spatialBins = py::cast<int>(value);
        } else if (key == "orientationBins") {
            params.orientationBins = py::cast<int>(value);
        } else if (key == "maxBinValue") {
            params.maxBinValue = py::cast<float>(value);
        } else if (key == "initialSigma") {
            params.initialSigma = py::cast<float>(value);
        } else if (key == "patchSize") {
            params.patchSize = py::cast<int>(value);
        } else if (key == "scale_min") {
            params.scale_min = py::cast<float>(value);
        } else if (key == "scale_max") {
            params.scale_max = py::cast<float>(value);
        } else if (key == "rotation_invariance") {
            params.rotation_invariance = py::cast<bool>(value);
        } else if (key == "augment_orientation") {
            params.augment_orientation = py::cast<bool>(value);
        } else if (key == "ori_maxima_thresh") {
            params.ori_maxima_thresh = py::cast<float>(value);
        } else if (key == "affine_invariance") {
            params.affine_invariance = py::cast<bool>(value);
        } else if (key == "only_count") {
            params.only_count = py::cast<bool>(value);
        } else if (key == "use_dense") {
            params.use_dense = py::cast<bool>(value);
        } else if (key == "dense_stride") {
            params.dense_stride = py::cast<int>(value);
        } else if (key == "siftPower") {
            params.siftPower = py::cast<float>(value);
        } else {
            py::print("[pyhesaff] WARNING: key=", key, " is not known");
        }
    }
}

static AffineHessianDetector* new_detector_from_fpath(const std::string& fpath, const HesaffParamsPy& params) {
    return new_hesaff_fpath(
        const_cast<char*>(fpath.c_str()),
        params.numberOfScales,
        params.threshold,
        params.edgeEigenValueRatio,
        params.border,
        params.maxPyramidLevels,
        params.maxIterations,
        params.convergenceThreshold,
        params.smmWindowSize,
        params.mrSize,
        params.spatialBins,
        params.orientationBins,
        params.maxBinValue,
        params.initialSigma,
        params.patchSize,
        params.scale_min,
        params.scale_max,
        params.rotation_invariance,
        params.augment_orientation,
        params.ori_maxima_thresh,
        params.affine_invariance,
        params.only_count,
        params.use_dense,
        params.dense_stride,
        params.siftPower);
}

static AffineHessianDetector* new_detector_from_image(const py::array& image, const HesaffParamsPy& params) {
    const auto buf = image.request();
    if (buf.ndim != 2 && buf.ndim != 3) {
        throw py::value_error("image must be 2D (grayscale) or 3D (color)");
    }
    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);
    int channels = 1;
    if (buf.ndim == 3) {
        channels = static_cast<int>(buf.shape[2]);
    }
    if (channels != 1 && channels != 3) {
        throw py::value_error("image must have 1 or 3 channels");
    }
    auto* data_ptr = static_cast<unsigned char*>(buf.ptr);
    return new_hesaff_image(
        data_ptr,
        rows,
        cols,
        channels,
        params.numberOfScales,
        params.threshold,
        params.edgeEigenValueRatio,
        params.border,
        params.maxPyramidLevels,
        params.maxIterations,
        params.convergenceThreshold,
        params.smmWindowSize,
        params.mrSize,
        params.spatialBins,
        params.orientationBins,
        params.maxBinValue,
        params.initialSigma,
        params.patchSize,
        params.scale_min,
        params.scale_max,
        params.rotation_invariance,
        params.augment_orientation,
        params.ori_maxima_thresh,
        params.affine_invariance,
        params.only_count,
        params.use_dense,
        params.dense_stride,
        params.siftPower);
}

static std::tuple<py::array, py::array> run_detect(DetectorHandle& detector) {
    int nKpts = 0;
    {
        py::gil_scoped_release release;
        nKpts = detect(detector.get());
    }
    const int kpts_dim = get_kpts_dim();
    const int desc_dim = get_desc_dim();
    auto kpts = py::array_t<float>({nKpts, kpts_dim});
    auto vecs = py::array_t<unsigned char>({nKpts, desc_dim});
    {
        py::gil_scoped_release release;
        exportArrays(detector.get(),
                     nKpts,
                     static_cast<float*>(kpts.mutable_data()),
                     static_cast<unsigned char*>(vecs.mutable_data()));
    }
    return std::make_tuple(kpts, vecs);
}

static py::tuple detect_fpath(const std::string& fpath, const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    auto result = run_detect(detector);
    return py::make_tuple(std::get<0>(result), std::get<1>(result));
}

static py::tuple detect_image(const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& image,
                              const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    auto result = run_detect(detector);
    return py::make_tuple(std::get<0>(result), std::get<1>(result));
}

static int count_image(const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& image,
                       const py::kwargs& kwargs) {
    HesaffParamsPy params;
    params.only_count = true;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    py::gil_scoped_release release;
    return detect(detector.get());
}

static py::array extract_desc_fpath(const std::string& fpath,
                                    const py::array_t<float, py::array::c_style | py::array::forcecast>& kpts,
                                    const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    const auto kpts_buf = kpts.request();
    if (kpts_buf.ndim != 2) {
        throw py::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts_buf.shape[0]);
    const int kpts_dim = static_cast<int>(kpts_buf.shape[1]);
    if (kpts_dim != get_kpts_dim()) {
        throw py::value_error("kpts have unexpected dimension");
    }
    auto vecs = py::array_t<unsigned char>({nKpts, get_desc_dim()});
    {
        py::gil_scoped_release release;
        extractDesc(detector.get(),
                    nKpts,
                    static_cast<float*>(kpts_buf.ptr),
                    static_cast<unsigned char*>(vecs.mutable_data()));
    }
    return vecs;
}

static py::array extract_desc_image(
    const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& kpts,
    const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    const auto kpts_buf = kpts.request();
    if (kpts_buf.ndim != 2) {
        throw py::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts_buf.shape[0]);
    const int kpts_dim = static_cast<int>(kpts_buf.shape[1]);
    if (kpts_dim != get_kpts_dim()) {
        throw py::value_error("kpts have unexpected dimension");
    }
    auto vecs = py::array_t<unsigned char>({nKpts, get_desc_dim()});
    {
        py::gil_scoped_release release;
        extractDesc(detector.get(),
                    nKpts,
                    static_cast<float*>(kpts_buf.ptr),
                    static_cast<unsigned char*>(vecs.mutable_data()));
    }
    return vecs;
}

static py::array extract_patches_fpath(const std::string& fpath,
                                       const py::array_t<float, py::array::c_style | py::array::forcecast>& kpts,
                                       const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    const auto kpts_buf = kpts.request();
    if (kpts_buf.ndim != 2) {
        throw py::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts_buf.shape[0]);
    const int kpts_dim = static_cast<int>(kpts_buf.shape[1]);
    if (kpts_dim != get_kpts_dim()) {
        throw py::value_error("kpts have unexpected dimension");
    }
    const int patch_size = params.patchSize;
    auto patches = py::array_t<float>({nKpts, patch_size, patch_size});
    {
        py::gil_scoped_release release;
        extractPatches(detector.get(),
                       nKpts,
                       static_cast<float*>(kpts_buf.ptr),
                       static_cast<float*>(patches.mutable_data()));
    }
    return patches;
}

static py::array extract_patches_image(
    const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& kpts,
    const py::kwargs& kwargs) {
    HesaffParamsPy params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    const auto kpts_buf = kpts.request();
    if (kpts_buf.ndim != 2) {
        throw py::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts_buf.shape[0]);
    const int kpts_dim = static_cast<int>(kpts_buf.shape[1]);
    if (kpts_dim != get_kpts_dim()) {
        throw py::value_error("kpts have unexpected dimension");
    }
    const int patch_size = params.patchSize;
    auto patches = py::array_t<float>({nKpts, patch_size, patch_size});
    {
        py::gil_scoped_release release;
        extractPatches(detector.get(),
                       nKpts,
                       static_cast<float*>(kpts_buf.ptr),
                       static_cast<float*>(patches.mutable_data()));
    }
    return patches;
}

static py::array extract_desc_from_patches(
    const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& patches) {
    const auto buf = patches.request();
    if (buf.ndim != 3) {
        throw py::value_error("patches must be a 3D array");
    }
    const int num_patches = static_cast<int>(buf.shape[0]);
    const int patch_h = static_cast<int>(buf.shape[1]);
    const int patch_w = static_cast<int>(buf.shape[2]);
    if (patch_h != patch_w) {
        throw py::value_error("patches must be square");
    }
    auto vecs = py::array_t<unsigned char>({num_patches, get_desc_dim()});
    {
        py::gil_scoped_release release;
        extractDescFromPatches(num_patches,
                               patch_h,
                               patch_w,
                               static_cast<unsigned char*>(buf.ptr),
                               static_cast<unsigned char*>(vecs.mutable_data()));
    }
    return vecs;
}

PYBIND11_MODULE(_hesaff, m) {
    m.doc() = "pybind11 bindings for pyhesaff";
    m.def("get_cpp_version", &get_cpp_version);
    m.def("is_debug_mode", []() { return static_cast<bool>(is_debug_mode()); });
    m.def("get_kpts_dim", &get_kpts_dim);
    m.def("get_desc_dim", &get_desc_dim);
    m.def("detect_fpath", &detect_fpath, py::arg("fpath"));
    m.def("detect_image", &detect_image, py::arg("image"));
    m.def("count_image", &count_image, py::arg("image"));
    m.def("extract_desc_fpath", &extract_desc_fpath, py::arg("fpath"), py::arg("kpts"));
    m.def("extract_desc_image", &extract_desc_image, py::arg("image"), py::arg("kpts"));
    m.def("extract_patches_fpath", &extract_patches_fpath, py::arg("fpath"), py::arg("kpts"));
    m.def("extract_patches_image", &extract_patches_image, py::arg("image"), py::arg("kpts"));
    m.def("extract_desc_from_patches", &extract_desc_from_patches, py::arg("patches"));
}
