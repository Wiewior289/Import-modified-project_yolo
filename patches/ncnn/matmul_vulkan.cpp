// SPDX-License-Identifier: BSD-3-Clause
// Conservative FP32 MatMul Vulkan backend.
// No cooperative-matrix, subgroup-shuffle, or FP16 arithmetic requirement.

#include "matmul_vulkan.h"

#include "layer_shader_type.h"
#include "pipeline.h"

#include <algorithm>

namespace ncnn {

namespace {

struct MatMulPlan
{
    int M = 1;
    int N = 1;
    int K = 1;

    int out_dims = 1;
    int out_w = 1;
    int out_h = 1;
    int out_d = 1;
    int out_c = 1;

    // 0 = matrix x matrix, 1 = vector x matrix/batched matrix,
    // 2 = matrix/batched matrix x vector, 3 = vector dot vector.
    int vector_mode = 0;

    // 2, 3, or 4: rank used to interpret batched input addressing.
    int batch_layout_dims = 2;
};

static bool broadcast_compatible(int a, int b)
{
    return a == b || a == 1 || b == 1;
}

static int normalized_d(const VkMat& m)
{
    if (m.dims == 4) return m.d;
    if (m.dims == 3) return m.c;
    return 1;
}

static int normalized_c(const VkMat& m)
{
    return m.dims == 4 ? m.c : 1;
}

static bool make_matmul_plan(const VkMat& A, const VkMat& B,
                             int transB, MatMulPlan& p)
{
    if (A.dims < 1 || A.dims > 4 || B.dims < 1 || B.dims > 4)
        return false;

    if (A.n != 1 || B.n != 1)
        return false;

    // NCNN's Vulkan graph may store activations as FP16. Keep the storage
    // type used by the surrounding graph, but accumulate in FP32 in the shader.
    if ((A.elemsize != 2u && A.elemsize != 4u) ||
        A.elemsize != B.elemsize ||
        A.elempack != 1 || B.elempack != 1)
        return false;

    p.K = A.w;
    const int b_k = B.dims == 1 ? B.w : (transB == 0 ? B.h : B.w);
    if (p.K <= 0 || b_k != p.K)
        return false;

    p.M = A.dims == 1 ? 1 : A.h;
    p.N = B.dims == 1 ? 1 : (transB == 0 ? B.w : B.h);

    if (p.M <= 0 || p.N <= 0)
        return false;

    const int ad = A.dims;
    const int bd = B.dims;
    const int max_dims = std::max(ad, bd);

    if (ad == 1 && bd == 1)
    {
        p.vector_mode = 3;
        p.out_dims = 1;
        p.out_w = 1;
        p.batch_layout_dims = 2;
        return true;
    }

    if (ad == 1 && bd == 2)
    {
        p.vector_mode = 1;
        p.out_dims = 1;
        p.out_w = p.N;
        p.batch_layout_dims = 2;
        return true;
    }

    if (ad == 2 && bd == 1)
    {
        p.vector_mode = 2;
        p.out_dims = 1;
        p.out_w = p.M;
        p.batch_layout_dims = 2;
        return true;
    }

    // Vector x batched matrix. NCNN flattens the (d,c) batch into output
    // rows for rank-3 input, and preserves (d,c) as (h,c) for rank-4 input.
    if (ad == 1 && bd > 2)
    {
        p.vector_mode = 1;
        p.batch_layout_dims = bd;
        if (bd == 3)
        {
            p.out_dims = 2;
            p.out_w = p.N;
            p.out_h = B.c;
        }
        else
        {
            p.out_dims = 3;
            p.out_w = p.N;
            p.out_h = B.d;
            p.out_c = B.c;
        }
        return p.out_h > 0 && p.out_c > 0;
    }

    // Batched matrix x vector.
    if (ad > 2 && bd == 1)
    {
        p.vector_mode = 2;
        p.batch_layout_dims = ad;
        if (ad == 3)
        {
            p.out_dims = 2;
            p.out_w = p.M;
            p.out_h = A.c;
        }
        else
        {
            p.out_dims = 3;
            p.out_w = p.M;
            p.out_h = A.d;
            p.out_c = A.c;
        }
        return p.out_h > 0 && p.out_c > 0;
    }

    if (max_dims == 2)
    {
        p.vector_mode = 0;
        p.batch_layout_dims = 2;
        p.out_dims = 2;
        p.out_w = p.N;
        p.out_h = p.M;
        return true;
    }

    if (max_dims == 3)
    {
        const int ac = ad == 3 ? A.c : 1;
        const int bc = bd == 3 ? B.c : 1;
        if (!broadcast_compatible(ac, bc))
            return false;

        p.vector_mode = 0;
        p.batch_layout_dims = 3;
        p.out_dims = 3;
        p.out_w = p.N;
        p.out_h = p.M;
        p.out_c = std::max(ac, bc);
        return p.out_c > 0;
    }

    // Rank-3 operands are promoted by the CPU layer to (w,h,old_c,1).
    // Therefore old c is the depth axis and c=1 after promotion.
    const int ad_batch_d = normalized_d(A);
    const int bd_batch_d = normalized_d(B);
    const int ad_batch_c = normalized_c(A);
    const int bd_batch_c = normalized_c(B);

    if (!broadcast_compatible(ad_batch_d, bd_batch_d) ||
        !broadcast_compatible(ad_batch_c, bd_batch_c))
        return false;

    p.vector_mode = 0;
    p.batch_layout_dims = 4;
    p.out_dims = 4;
    p.out_w = p.N;
    p.out_h = p.M;
    p.out_d = std::max(ad_batch_d, bd_batch_d);
    p.out_c = std::max(ad_batch_c, bd_batch_c);
    return p.out_d > 0 && p.out_c > 0;
}

} // namespace

MatMul_vulkan::MatMul_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
    pipeline_matmul = 0;
}

int MatMul_vulkan::create_pipeline(const Option& opt)
{
    delete pipeline_matmul;
    pipeline_matmul = 0;

    // Preserve NCNN's FP16/FP32 storage ABI so this layer can connect to
    // either kind of Vulkan producer without a hidden conversion/download.
    // Arithmetic remains FP32 for accuracy and to avoid implicit-fp16 quirks.
    Option matmul_opt = opt;
    matmul_opt.use_fp16_arithmetic = false;
    matmul_opt.use_fp16_uniform = false;

    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = transB;

    pipeline_matmul = new Pipeline(vkdev);
    pipeline_matmul->set_local_size_xyz(8, 8, 1);

    const int ret = pipeline_matmul->create(
        LayerShaderType::matmul, matmul_opt, specializations);
    if (ret != 0)
    {
        delete pipeline_matmul;
        pipeline_matmul = 0;
        return ret;
    }

    return 0;
}

int MatMul_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_matmul;
    pipeline_matmul = 0;
    return 0;
}

int MatMul_vulkan::forward(const std::vector<VkMat>& bottom_blobs,
                           std::vector<VkMat>& top_blobs,
                           VkCompute& cmd, const Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 1 || !pipeline_matmul)
        return -1;

    const VkMat& A = bottom_blobs[0];
    const VkMat& B = bottom_blobs[1];
    VkMat& top_blob = top_blobs[0];

    MatMulPlan plan;
    if (!make_matmul_plan(A, B, transB, plan))
    {
        NCNN_LOGE("MatMul_vulkan unsupported shape/storage A=%d:%d,%d,%d,%d pack=%d es=%zu B=%d:%d,%d,%d,%d pack=%d es=%zu transB=%d",
                  A.dims, A.w, A.h, A.d, A.c, A.elempack, A.elemsize,
                  B.dims, B.w, B.h, B.d, B.c, B.elempack, B.elemsize,
                  transB);
        return -1;
    }

    if (plan.out_dims == 1)
        top_blob.create(plan.out_w, A.elemsize, opt.blob_vkallocator);
    else if (plan.out_dims == 2)
        top_blob.create(plan.out_w, plan.out_h, A.elemsize, opt.blob_vkallocator);
    else if (plan.out_dims == 3)
        top_blob.create(plan.out_w, plan.out_h, plan.out_c, A.elemsize, opt.blob_vkallocator);
    else
        top_blob.create(plan.out_w, plan.out_h, plan.out_d, plan.out_c,
                        A.elemsize, opt.blob_vkallocator);

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = A;
    bindings[1] = B;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(23);
    constants[0].i = plan.M;
    constants[1].i = plan.N;
    constants[2].i = plan.K;

    constants[3].i = A.dims;
    constants[4].i = A.w;
    constants[5].i = A.h;
    constants[6].i = A.d;
    constants[7].i = A.c;
    constants[8].i = (int)A.cstep;

    constants[9].i = B.dims;
    constants[10].i = B.w;
    constants[11].i = B.h;
    constants[12].i = B.d;
    constants[13].i = B.c;
    constants[14].i = (int)B.cstep;

    constants[15].i = top_blob.dims;
    constants[16].i = top_blob.w;
    constants[17].i = top_blob.h;
    constants[18].i = top_blob.d;
    constants[19].i = top_blob.c;
    constants[20].i = (int)top_blob.cstep;

    constants[21].i = plan.vector_mode;
    constants[22].i = plan.batch_layout_dims;

    // The real output VkMat is the dispatcher. In particular, its d/c fields
    // are required for complete 4-D dispatch coverage.
    cmd.record_pipeline(pipeline_matmul, bindings, constants, top_blob);
    return 0;
}

} // namespace ncnn
