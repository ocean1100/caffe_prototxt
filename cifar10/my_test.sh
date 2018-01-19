#!/usr/bin/env sh
set -e

CAFFEHOME=/home/ocean1101/Workspace/ssd/caffe
TOOLS=$CAFFEHOME/build/tools

#$TOOLS/caffe train \
#    --solver=cifar10/solver.prototxt $@

$TOOLS/caffe test \
    --gpu=all \
    --model=cifar10/train_val.prototxt   \
    --weights=alex_adam_iter_10000.caffemodel   \
    --iterations=100
#    --solver=cifar10/solver_test.prototxt \
#    --snapshot=_iter_33526.solverstate $@
