// Phase 1b NPU launch smoke — runs on the GitCode Space 910B container.
//
// What it proves: the cross-built device blob registers with the
// ascendc runtime, the auto-generated `aclrtlaunch_*` API actually
// reaches the NPU, and `aclrtSynchronizeStream` returns. The kernel
// itself is a no-op placeholder; numerics are out of scope here.
//
// Stdout is what the Gradio app pipes back to the page.

#include <acl/acl.h>
#include <cstdio>
#include <cstring>

#include "gemm_w4a4.h"

static const char* AclErr() {
    const char* s = aclGetRecentErrMsg();
    return (s && *s) ? s : "(no aclGetRecentErrMsg)";
}

int main() {
    std::printf("[smoke] aclInit\n");
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed: %s\n", AclErr());
        return 1;
    }

    int32_t deviceCount = 0;
    aclrtGetDeviceCount(reinterpret_cast<uint32_t*>(&deviceCount));
    std::printf("[smoke] device count: %d\n", deviceCount);
    if (deviceCount <= 0) {
        std::fprintf(stderr, "no NPU devices visible\n");
        aclFinalize();
        return 2;
    }

    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(0) failed: %s\n", AclErr());
        aclFinalize();
        return 3;
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateStream failed: %s\n", AclErr());
        aclrtResetDevice(0);
        aclFinalize();
        return 4;
    }

    // Placeholder params. The Phase 1b device kernel doesn't dereference,
    // but kernel.cpp still does the H2D copy unconditionally.
    svdquant::GemmW4A4Params p{};
    std::printf("[smoke] launching gemm_w4a4 (placeholder kernel, no-op)\n");
    svdquant::ascend::gemm_w4a4(p, stream);

    std::printf("[smoke] aclrtSynchronizeStream\n");
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSynchronizeStream failed (%d): %s\n",
                     static_cast<int>(sync_ret), AclErr());
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 5;
    }

    std::printf("[smoke] OK -- kernel launched and stream synchronized\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
