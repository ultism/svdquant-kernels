# Each subdir is a "kernel pod": self-contained sources for one SVDQuant op.
# Layout:
#   include/<name>.h          — public C/C++ header
#   cuda/kernel.cu            — (optional) CUDA C++ kernel (currently unused;
#                               CUDA path is Python CuTe DSL, not nvcc)
#   ascend/kernel.cpp         — (optional) host launcher, plain C++
#   ascend/kernel_device.cpp  — (optional) __aicore__ device source, ccec-compiled
#                               via CANN's ascendc_library()
#
# When kernel_device.cpp is present and ascendc.cmake is available, the helper
# builds it as a separate `<name>_device` ascendc static lib and links it into
# the pod's host OBJECT lib so consumers don't have to know the difference.
set(PTO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/pto-isa/include" CACHE PATH "PTO ISA include path")

function(svdquant_add_kernel_pod name)
    set(_root "${CMAKE_CURRENT_LIST_DIR}/${name}")
    set(_sources "")
    if(SVDQUANT_ENABLE_CUDA AND EXISTS "${_root}/cuda/kernel.cu")
        list(APPEND _sources "${_root}/cuda/kernel.cu")
    endif()
    if(SVDQUANT_ENABLE_ASCEND AND EXISTS "${_root}/ascend/kernel.cpp")
        list(APPEND _sources "${_root}/ascend/kernel.cpp")
    endif()
    if(NOT _sources)
        message(STATUS "svdquant: skipping kernel pod '${name}' (no enabled sources)")
        return()
    endif()

    add_library(svdquant_${name} OBJECT ${_sources})
    target_link_libraries(svdquant_${name} PUBLIC svdquant::common)
    target_include_directories(svdquant_${name} PUBLIC "${_root}/include")
    if(SVDQUANT_ENABLE_ASCEND)
        target_include_directories(svdquant_${name} PRIVATE ${CANN_INCLUDE_DIRS})
    endif()
    add_library(svdquant::${name} ALIAS svdquant_${name})

    # ---- ascendc cross-compile of __aicore__ device source ----
    # The /./ path-disambiguator that CMake injects when a source is
    # outside the sub-build's CMAKE_SOURCE_DIR is normalized away by
    # our ascendc_python_wrapper.py (set as ASCEND_PYTHON_EXECUTABLE
    # in the top-level CMakeLists). With that fix in place we can
    # pass the device source by its true path.
    set(_dev_src "${_root}/ascend/kernel_device.cpp")
    if(SVDQUANT_ENABLE_ASCEND AND SVDQUANT_ASCENDC_AVAILABLE AND EXISTS "${_dev_src}")
        ascendc_library(svdquant_${name}_device STATIC "${_dev_src}")
        ascendc_include_directories(svdquant_${name}_device PRIVATE
            "${PTO_INCLUDE_DIR}"
            "${PTO_INCLUDE_DIR}/common"
            # The device source can include the pod's public C header
            # (for the params struct definition) and svdquant/tensor.h
            # transitively. We mirror the host OBJECT lib's include
            # tree here since ascendc_include_directories doesn't
            # inherit from the host target.
            "${_root}/include"
            "${CMAKE_SOURCE_DIR}/csrc/common/include"
        )
        # Host OBJECT lib pulls in the ascendc-built device static lib
        # so downstream linkers don't need to know about it.
        target_link_libraries(svdquant_${name} PUBLIC svdquant_${name}_device)
        message(STATUS "svdquant: kernel pod '${name}' includes ascendc device source")
    endif()
endfunction()

svdquant_add_kernel_pod(gemm_w4a4)

# Preprocessing op `quantize_w4a4_act_fuse_lora` is memory-bound and
# needs to run on both CUDA and Ascend, so it lives in the Triton
# directory (`triton_kernels/`) instead of here — Triton's Python +
# LLVM toolchain isn't driven by CMake. See `triton_kernels/README.md`
# for the library-choice rationale.
