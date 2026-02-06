#include <cmath>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "hesaff_capi.h"

namespace nb = nanobind;

struct HesaffParamsNb {
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

static void update_params_from_kwargs(HesaffParamsNb& params, const nb::kwargs& kwargs) {
    for (const auto& item : kwargs) {
        const std::string key = nb::cast<std::string>(item.first);
        const nb::handle value = item.second;
        if (key == "numberOfScales") {
            params.numberOfScales = nb::cast<int>(value);
        } else if (key == "threshold") {
            params.threshold = nb::cast<float>(value);
        } else if (key == "edgeEigenValueRatio") {
            params.edgeEigenValueRatio = nb::cast<float>(value);
        } else if (key == "border") {
            params.border = nb::cast<int>(value);
        } else if (key == "maxPyramidLevels") {
            params.maxPyramidLevels = nb::cast<int>(value);
        } else if (key == "maxIterations") {
            params.maxIterations = nb::cast<int>(value);
        } else if (key == "convergenceThreshold") {
            params.convergenceThreshold = nb::cast<float>(value);
        } else if (key == "smmWindowSize") {
            params.smmWindowSize = nb::cast<int>(value);
        } else if (key == "mrSize") {
            params.mrSize = nb::cast<float>(value);
        } else if (key == "spatialBins") {
            params.spatialBins = nb::cast<int>(value);
        } else if (key == "orientationBins") {
            params.orientationBins = nb::cast<int>(value);
        } else if (key == "maxBinValue") {
            params.maxBinValue = nb::cast<float>(value);
        } else if (key == "initialSigma") {
            params.initialSigma = nb::cast<float>(value);
        } else if (key == "patchSize") {
            params.patchSize = nb::cast<int>(value);
        } else if (key == "scale_min") {
            params.scale_min = nb::cast<float>(value);
        } else if (key == "scale_max") {
            params.scale_max = nb::cast<float>(value);
        } else if (key == "rotation_invariance") {
            params.rotation_invariance = nb::cast<bool>(value);
        } else if (key == "augment_orientation") {
            params.augment_orientation = nb::cast<bool>(value);
        } else if (key == "ori_maxima_thresh") {
            params.ori_maxima_thresh = nb::cast<float>(value);
        } else if (key == "affine_invariance") {
            params.affine_invariance = nb::cast<bool>(value);
        } else if (key == "only_count") {
            params.only_count = nb::cast<bool>(value);
        } else if (key == "use_dense") {
            params.use_dense = nb::cast<bool>(value);
        } else if (key == "dense_stride") {
            params.dense_stride = nb::cast<int>(value);
        } else if (key == "siftPower") {
            params.siftPower = nb::cast<float>(value);
        } else {
            nb::print("[pyhesaff] WARNING: key=", key, " is not known");
        }
    }
}

static AffineHessianDetector* new_detector_from_fpath(const std::string& fpath, const HesaffParamsNb& params) {
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

static AffineHessianDetector* new_detector_from_image(
    const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& image,
    const HesaffParamsNb& params) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw nb::value_error("image must be 2D (grayscale) or 3D (color)");
    }
    const int rows = static_cast<int>(image.shape(0));
    const int cols = static_cast<int>(image.shape(1));
    int channels = 1;
    if (image.ndim() == 3) {
        channels = static_cast<int>(image.shape(2));
    }
    if (channels != 1 && channels != 3) {
        throw nb::value_error("image must have 1 or 3 channels");
    }
    auto* data_ptr = static_cast<uint8_t*>(image.data());
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

static std::pair<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, uint8_t>> run_detect(DetectorHandle& detector) {
    int nKpts = 0;
    {
        nb::gil_scoped_release release;
        nKpts = detect(detector.get());
    }
    const int kpts_dim = get_kpts_dim();
    const int desc_dim = get_desc_dim();
    nb::ndarray<nb::numpy, float> kpts({static_cast<size_t>(nKpts), static_cast<size_t>(kpts_dim)});
    nb::ndarray<nb::numpy, uint8_t> vecs({static_cast<size_t>(nKpts), static_cast<size_t>(desc_dim)});
    {
        nb::gil_scoped_release release;
        exportArrays(detector.get(),
                     nKpts,
                     static_cast<float*>(kpts.data()),
                     static_cast<uint8_t*>(vecs.data()));
    }
    return {kpts, vecs};
}

static nb::tuple detect_fpath(const std::string& fpath, const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    auto result = run_detect(detector);
    return nb::make_tuple(result.first, result.second);
}

static nb::tuple detect_image(const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& image, const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    auto result = run_detect(detector);
    return nb::make_tuple(result.first, result.second);
}

static int count_image(const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& image, const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    params.only_count = true;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    nb::gil_scoped_release release;
    return detect(detector.get());
}

static nb::ndarray<nb::numpy, uint8_t> extract_desc_fpath(
    const std::string& fpath,
    const nb::ndarray<nb::numpy, float, nb::c_contig>& kpts,
    const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    if (kpts.ndim() != 2) {
        throw nb::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts.shape(0));
    const int kpts_dim = static_cast<int>(kpts.shape(1));
    if (kpts_dim != get_kpts_dim()) {
        throw nb::value_error("kpts have unexpected dimension");
    }
    nb::ndarray<nb::numpy, uint8_t> vecs({static_cast<size_t>(nKpts), static_cast<size_t>(get_desc_dim())});
    {
        nb::gil_scoped_release release;
        extractDesc(detector.get(),
                    nKpts,
                    static_cast<float*>(kpts.data()),
                    static_cast<uint8_t*>(vecs.data()));
    }
    return vecs;
}

static nb::ndarray<nb::numpy, uint8_t> extract_desc_image(
    const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& image,
    const nb::ndarray<nb::numpy, float, nb::c_contig>& kpts,
    const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    if (kpts.ndim() != 2) {
        throw nb::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts.shape(0));
    const int kpts_dim = static_cast<int>(kpts.shape(1));
    if (kpts_dim != get_kpts_dim()) {
        throw nb::value_error("kpts have unexpected dimension");
    }
    nb::ndarray<nb::numpy, uint8_t> vecs({static_cast<size_t>(nKpts), static_cast<size_t>(get_desc_dim())});
    {
        nb::gil_scoped_release release;
        extractDesc(detector.get(),
                    nKpts,
                    static_cast<float*>(kpts.data()),
                    static_cast<uint8_t*>(vecs.data()));
    }
    return vecs;
}

static nb::ndarray<nb::numpy, float> extract_patches_fpath(
    const std::string& fpath,
    const nb::ndarray<nb::numpy, float, nb::c_contig>& kpts,
    const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_fpath(fpath, params));
    if (kpts.ndim() != 2) {
        throw nb::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts.shape(0));
    const int kpts_dim = static_cast<int>(kpts.shape(1));
    if (kpts_dim != get_kpts_dim()) {
        throw nb::value_error("kpts have unexpected dimension");
    }
    const int patch_size = params.patchSize;
    nb::ndarray<nb::numpy, float> patches({static_cast<size_t>(nKpts),
                                           static_cast<size_t>(patch_size),
                                           static_cast<size_t>(patch_size)});
    {
        nb::gil_scoped_release release;
        extractPatches(detector.get(),
                       nKpts,
                       static_cast<float*>(kpts.data()),
                       static_cast<float*>(patches.data()));
    }
    return patches;
}

static nb::ndarray<nb::numpy, float> extract_patches_image(
    const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& image,
    const nb::ndarray<nb::numpy, float, nb::c_contig>& kpts,
    const nb::kwargs& kwargs) {
    HesaffParamsNb params;
    update_params_from_kwargs(params, kwargs);
    DetectorHandle detector(new_detector_from_image(image, params));
    if (kpts.ndim() != 2) {
        throw nb::value_error("kpts must be a 2D array");
    }
    const int nKpts = static_cast<int>(kpts.shape(0));
    const int kpts_dim = static_cast<int>(kpts.shape(1));
    if (kpts_dim != get_kpts_dim()) {
        throw nb::value_error("kpts have unexpected dimension");
    }
    const int patch_size = params.patchSize;
    nb::ndarray<nb::numpy, float> patches({static_cast<size_t>(nKpts),
                                           static_cast<size_t>(patch_size),
                                           static_cast<size_t>(patch_size)});
    {
        nb::gil_scoped_release release;
        extractPatches(detector.get(),
                       nKpts,
                       static_cast<float*>(kpts.data()),
                       static_cast<float*>(patches.data()));
    }
    return patches;
}

static nb::ndarray<nb::numpy, uint8_t> extract_desc_from_patches(
    const nb::ndarray<nb::numpy, uint8_t, nb::c_contig>& patches) {
    if (patches.ndim() != 3) {
        throw nb::value_error("patches must be a 3D array");
    }
    const int num_patches = static_cast<int>(patches.shape(0));
    const int patch_h = static_cast<int>(patches.shape(1));
    const int patch_w = static_cast<int>(patches.shape(2));
    if (patch_h != patch_w) {
        throw nb::value_error("patches must be square");
    }
    nb::ndarray<nb::numpy, uint8_t> vecs({static_cast<size_t>(num_patches),
                                          static_cast<size_t>(get_desc_dim())});
    {
        nb::gil_scoped_release release;
        extractDescFromPatches(num_patches,
                               patch_h,
                               patch_w,
                               static_cast<uint8_t*>(patches.data()),
                               static_cast<uint8_t*>(vecs.data()));
    }
    return vecs;
}

NB_MODULE(_hesaff, m) {
    m.doc() = "nanobind bindings for pyhesaff";
    m.def("get_cpp_version", &get_cpp_version);
    m.def("is_debug_mode", []() { return static_cast<bool>(is_debug_mode()); });
    m.def("get_kpts_dim", &get_kpts_dim);
    m.def("get_desc_dim", &get_desc_dim);
    m.def("detect_fpath", &detect_fpath, nb::arg("fpath"), nb::arg("kwargs") = nb::kwargs());
    m.def("detect_image", &detect_image, nb::arg("image"), nb::arg("kwargs") = nb::kwargs());
    m.def("count_image", &count_image, nb::arg("image"), nb::arg("kwargs") = nb::kwargs());
    m.def("extract_desc_fpath", &extract_desc_fpath, nb::arg("fpath"), nb::arg("kpts"), nb::arg("kwargs") = nb::kwargs());
    m.def("extract_desc_image", &extract_desc_image, nb::arg("image"), nb::arg("kpts"), nb::arg("kwargs") = nb::kwargs());
    m.def("extract_patches_fpath", &extract_patches_fpath, nb::arg("fpath"), nb::arg("kpts"), nb::arg("kwargs") = nb::kwargs());
    m.def("extract_patches_image", &extract_patches_image, nb::arg("image"), nb::arg("kpts"), nb::arg("kwargs") = nb::kwargs());
    m.def("extract_desc_from_patches", &extract_desc_from_patches, nb::arg("patches"));
}
