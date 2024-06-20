# octree-builder
A CUDA-implemented octree-builder module.

## Dependencies
This code has only been built and tested under `Ubuntu-20.04` with `RTX 3090` GPU. Other dependencies:

```
python                    3.7.16
pytorch                   1.12.1
```

Note that other versions of `Python` and `PyTorch` are also likely to run `octree-builder` well.

## Installation
```
$ git clone https://github.com/ZhiyeTang/octree-builder.git
$ cd octree-builder
$ pip install .
```

## Usage
```{python}
from octree_builder._C import buildOctree

tree = buildOctree(grid_coord, level)
```

in which `grid_coord` is the spatial indices of all nodes within the octree, with a shape like `[N, 3]`, while `level` is the levels of each node with a shape like `[N, 1]` or `[N]`.