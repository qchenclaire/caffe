#!/usr/bin/env sh
set -e

TOOLS=./build/tools
# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/resnet_18/resnet_18_solver.prototxt  $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/resnet_18/resnet_18_solver_lr1.prototxt \
    --snapshot=examples/cifar10/resnet_18/base_iter_60000.solverstate.h5 $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/resnet_18/resnet_18_solver_lr2.prototxt \
    --snapshot=examples/cifar10/resnet_18/base_iter_65000.solverstate.h5 $@
