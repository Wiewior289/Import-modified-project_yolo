android {
    namespace = "com.tencent.yolo11ncnn"
    compileSdk = 34
	ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.tencent.yolo11ncnn"
        minSdk = 26
        targetSdk = 34

        externalNativeBuild {
            cmake {
                // Włączenie wsparcia dla instrukcji C++17 i Vulkana
                cppFlags("-std=c++17")
                arguments("-DANDROID_STL=c++_shared")
            }
        }
        
        ndk {
            // Kompilacja dla architektur ARM (S9+ to arm64-v8a)
            abiFilters.addAll(listOf("arm64-v8a"))
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }
}