/*
 * Copyright (C) 2024, Shenzhen University
 * Immersive Media Laboratory, Institute of Future Media Computing
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use.
 */

#include <cooperative_groups.h>
#include "octree_builder.h"

#define BRANCH 2
#define THREADS 256

__device__ bool isEq3D(float4 p, float4 q)
{
    return (p.x == q.x && p.y == q.y && p.z == q.z);
}

__device__ bool isEq3D(float3 p, float3 q)
{
    return (p.x == q.x && p.y == q.y && p.z == q.z);
}

__device__ bool isEq3D(int3 p, int3 q)
{
    return (p.x == q.x && p.y == q.y && p.z == q.z);
}

__global__ void catchChildren(const int3* grid_coords, const int* levels, const int P, int* rst, const int lv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P || idx < 0) return;
    if (levels[idx] != lv) return;
    
    const int parent_level = lv - 1;

    int3 anchor = grid_coords[idx];

    int3 parent = {anchor.x / BRANCH, anchor.y / BRANCH, anchor.z / BRANCH};

    for (int i = 0; i < P; i++)
    {
        if (isEq3D(grid_coords[i], parent) && levels[i] == parent_level)
        {
            rst[idx] = i;
            return;
        }
    }
}

torch::Tensor buildOctree(const torch::Tensor& grid_coords, const torch::Tensor& levels)
{

    // catch cuda error
    cudaError_t cudaStatus;

    const int P = grid_coords.size(0);
    const int levelMin = torch::min(levels).item().toInt();
    const int levelMax = torch::max(levels).item().toInt();

    // initialize `rst` for result
    auto rst_opts = grid_coords.options().dtype(torch::kInt32);
    torch::Tensor rst = torch::full({P}, -1, rst_opts);

    if (rst.device() != grid_coords.device() || grid_coords.device() != levels.device())
    {
        throw "Tensors should be on a same device.\n";
    }
    else
    {
        std::cout << "Starting Octree Matching" << std::endl;
    }

    // parallel computing
    int3* grid_coord_ptr;
    cudaMalloc(&grid_coord_ptr, grid_coords.numel() * sizeof(int));
    cudaMemcpy(grid_coord_ptr, grid_coords.contiguous().data_ptr<int>(), grid_coords.numel() * sizeof(int), cudaMemcpyHostToDevice);

    int* level_ptr;
    cudaMalloc(&level_ptr, levels.numel() * sizeof(int));
    cudaMemcpy(level_ptr, levels.contiguous().data_ptr<int>(), levels.numel() * sizeof(int), cudaMemcpyHostToDevice);
    int* rst_ptr = rst.contiguous().data_ptr<int>();
    cudaDeviceSynchronize();

    int BLOCKS = (P + THREADS - 1) / THREADS;
    for (int lv = levelMax; lv > levelMin; lv--)
    {
        catchChildren<<<BLOCKS, THREADS>>>(grid_coord_ptr, level_ptr, P, rst_ptr, lv);
        cudaDeviceSynchronize();
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();

    return rst;
}