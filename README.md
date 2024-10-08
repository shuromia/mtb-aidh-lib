# mtb-aidh-lib

Usage

Adding the Library
To use the library in ModusToolbox v3.x, add a manifest.loc file to the .modustoolbox folder in your home directory. Inside this file, paste the following URI:
     
        https://github.com/shuromia/mtb-tflm-lib/raw/main/manifests/mtb-super-manifest-supplement-aidh.xml
                
Quick Start

To add this library to a ModusToolbox project, install it from the Library Manager. Then add the following DEFINES, COMPONENTS, and CXXFLAGS to the application's Makefile:

        DEFINES+= TF_LITE_STATIC_MEMORY        
        COMPONENTS+= TFLM        
        CXXFLAGS+= -std=c++11
