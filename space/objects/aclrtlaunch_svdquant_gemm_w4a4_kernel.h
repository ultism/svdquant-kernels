#ifndef HEADER_ACLRTLAUNCH_SVDQUANT_GEMM_W4A4_KERNEL_H
#define HEADER_ACLRTLAUNCH_SVDQUANT_GEMM_W4A4_KERNEL_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_svdquant_gemm_w4a4_kernel(uint32_t blockDim, aclrtStream stream, void* params_addr);
#endif
