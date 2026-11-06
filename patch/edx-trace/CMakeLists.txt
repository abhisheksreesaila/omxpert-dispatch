set(TARGET llama-edx-trace)
add_executable(${TARGET} edx-trace.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
