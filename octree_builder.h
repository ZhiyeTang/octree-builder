/*
 * Copyright (C) 2024, Shenzhen University
 * Immersive Media Laboratory, Institute of Future Media Computing
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use.
 */

#include <torch/extension.h>
#ifndef OCTREEBUILDER_H_INCLUDED
#define OCTREEBUILDER_H_INCLUDED

// OctreeNode* clusterSiblings(float3 anchor, const float3* siblings, int P, float voxelSize);

torch::Tensor buildOctree(const torch::Tensor& grid_coords, const torch::Tensor& levels);

#endif