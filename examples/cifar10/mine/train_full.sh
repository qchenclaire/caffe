#!/usr/bin/env sh
#set -e
rm examples/cifar10/mine/log
cp examples/cifar10/cifar10_pool_25000/bins_init.p examples/cifar10/cifar10_pool_25000/bins.p
TOOLS=./build/tools
python python/rewrite.py 36000
$TOOLS/caffe train \
    --solver=examples/cifar10/mine/cifar10_full_solver.prototxt \
    --snapshot=examples/cifar10/cifar10_full/_iter_36000.solverstate.h5 \
    -gpu 0 2>&1 | tee -a examples/cifar10/mine/log $@
python python/rewrite.py 56000
# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/mine/cifar10_full_solver_lr1.prototxt \
    --snapshot=examples/cifar10/mine/models/_iter_56000.solverstate.h5 \
    -gpu 0 2>&1 | tee -a examples/cifar10/mine/log $@
python python/rewrite.py 63000
# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/mine/cifar10_full_solver_lr2.prototxt \
    --snapshot=examples/cifar10/mine/models/_iter_63000.solverstate.h5 \
    -gpu 0 2>&1 | tee -a examples/cifar10/mine/log  $@



