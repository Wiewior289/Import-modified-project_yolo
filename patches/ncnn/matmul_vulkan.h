// SPDX-License-Identifier: BSD-3-Clause
// App-local NCNN extension. The upstream MatMul CPU implementation remains intact.

#ifndef LAYER_MATMUL_VULKAN_H
#define LAYER_MATMUL_VULKAN_H

#include "matmul.h"

namespace ncnn {

class MatMul_vulkan : public MatMul
{
public:
    MatMul_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using MatMul::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs,
                        std::vector<VkMat>& top_blobs,
                        VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_matmul;
};

} // namespace ncnn

#endif // LAYER_MATMUL_VULKAN_H
