#ifndef HESAFF_CAPI_H
#define HESAFF_CAPI_H

#include "hesaff.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AffineHessianDetector;

HESAFF_EXPORTED int detect(AffineHessianDetector* detector);
HESAFF_EXPORTED int get_cpp_version();
HESAFF_EXPORTED int is_debug_mode();
HESAFF_EXPORTED int get_kpts_dim();
HESAFF_EXPORTED int get_desc_dim();

HESAFF_EXPORTED AffineHessianDetector* new_hesaff_image(
    unsigned char* imgin,
    int rows,
    int cols,
    int channels,
    int numberOfScales,
    float threshold,
    float edgeEigenValueRatio,
    int border,
    int maxPyramidLevels,
    int maxIterations,
    float convergenceThreshold,
    int smmWindowSize,
    float mrSize,
    int spatialBins,
    int orientationBins,
    float maxBinValue,
    float initialSigma,
    int patchSize,
    float scale_min,
    float scale_max,
    bool rotation_invariance,
    bool augment_orientation,
    float ori_maxima_thresh,
    bool affine_invariance,
    bool only_count,
    bool use_dense,
    int dense_stride,
    float siftPower);

HESAFF_EXPORTED AffineHessianDetector* new_hesaff_fpath(
    char* img_fpath,
    int numberOfScales,
    float threshold,
    float edgeEigenValueRatio,
    int border,
    int maxPyramidLevels,
    int maxIterations,
    float convergenceThreshold,
    int smmWindowSize,
    float mrSize,
    int spatialBins,
    int orientationBins,
    float maxBinValue,
    float initialSigma,
    int patchSize,
    float scale_min,
    float scale_max,
    bool rotation_invariance,
    bool augment_orientation,
    float ori_maxima_thresh,
    bool affine_invariance,
    bool only_count,
    bool use_dense,
    int dense_stride,
    float siftPower);

HESAFF_EXPORTED AffineHessianDetector* new_hesaff_imgpath_noparams(char* img_fpath);

HESAFF_EXPORTED void free_hesaff(AffineHessianDetector* detector);
HESAFF_EXPORTED void extractDesc(AffineHessianDetector* detector, int nKpts, float* kpts, unsigned char* desc);
HESAFF_EXPORTED void extractPatches(AffineHessianDetector* detector, int nKpts, float* kpts, float* patch_array);
HESAFF_EXPORTED void exportArrays(AffineHessianDetector* detector, int nKpts, float* kpts, unsigned char* desc);
HESAFF_EXPORTED void writeFeatures(AffineHessianDetector* detector, char* img_fpath);
HESAFF_EXPORTED void extractDescFromPatches(int num_patches, int patch_h, int patch_w, unsigned char* patches_array, unsigned char* descriptors_array);

#ifdef __cplusplus
}
#endif

#endif
