#include <cstdio>
#include <iostream>

#include "hesaff_capi.h"

int main(int argc, char** argv) {
    const char* about_message =
        "\nUsage: hesaffexe image_name.png\nDescribes elliptical keypoints (with gravity vector) given in "
        "kpts_file.txt using a SIFT descriptor. The help message has unfortunately been deleted. Check github "
        "history for details. https://github.com/perdoch/hesaff/blob/master/hesaff.cpp\n\n";

    if (argc > 1) {
        char* img_fpath = argv[1];
        AffineHessianDetector* detector = new_hesaff_imgpath_noparams(img_fpath);
        int nKpts = detect(detector);
        writeFeatures(detector, img_fpath);
        std::cout << "[main] nKpts: " << nKpts << std::endl;
        free_hesaff(detector);
    } else {
        std::printf("%s", about_message);
    }
    return 0;
}
