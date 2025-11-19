#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_d2xcp0_sfm_1vc_1ocv_MainActivity_nativeTest(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Native C++ is working!";
    return env->NewStringUTF(hello.c_str());
}