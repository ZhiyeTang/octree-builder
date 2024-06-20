/*
 * Copyright (C) 2024, Shenzhen University
 * Immersive Media Laboratory, Institute of Future Media Computing
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use.
 */

#include <torch/extension.h>
#include "octree_builder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buildOctree", &buildOctree);
}
