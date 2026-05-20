#ifndef HEADER_ACLRTLAUNCH_HELLOWORLD_H
#define HEADER_ACLRTLAUNCH_HELLOWORLD_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_helloworld(uint32_t blockDim, aclrtStream stream, void* x, void* y, void* z, uint32_t totalLength);
#endif
