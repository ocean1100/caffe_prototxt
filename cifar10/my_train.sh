#!/usr/bin/env sh
set -e

CAFFEHOME=/home/ocean1101/Workspace/ssd/caffe
TOOLS=$CAFFEHOME/build/tools

#$TOOLS/caffe train \
#    --solver=cifar10/solver.prototxt $@

$TOOLS/caffe train \
    --gpu=all \
    --solver=cifar10/solver.prototxt \
    --snapshot=alex_adam_iter_15000.solverstate $@
