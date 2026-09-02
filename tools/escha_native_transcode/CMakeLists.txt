cmake_minimum_required(VERSION 3.20)
project(escha_native_transcode LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

add_executable(escha-native-transcode main.cpp)
target_compile_features(escha-native-transcode PRIVATE cxx_std_20)
target_compile_options(escha-native-transcode PRIVATE
    -O3 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
target_link_libraries(escha-native-transcode PRIVATE OpenMP::OpenMP_CXX OpenSSL::Crypto dl)

