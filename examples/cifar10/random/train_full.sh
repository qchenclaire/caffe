#!/usr/bin/env sh
set -e
rm examples/cifar10/random/log
TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/random/cifar10_full_solver.prototxt \
    --snapshot=examples/cifar10/cifar10_full/_iter_30000.solverstate.h5 \
    -gpu 2 2>&1 | tee -a examples/cifar10/random/log $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/random/cifar10_full_solver_lr1.prototxt \
    --snapshot=examples/cifar10/random/models/_iter_56000.solverstate.h5 \
    -gpu 2 2>&1 | tee -a examples/cifar10/random/log $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/random/cifar10_full_solver_lr2.prototxt \
    --snapshot=examples/cifar10/random/models/_iter_65000.solverstate.h5 \
    -gpu 2 2>&1 | tee -a examples/cifar10/random/log  $@

