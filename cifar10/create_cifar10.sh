#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e

CAFFEHOME=/home/ocean1101/Workspace/ssd/caffe
DATA=data/cifar10
OUTPUT=data/cifar10_lmdb
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $OUTPUT/cifar10_train_$DBTYPE $OUTPUT/cifar10_validate_$DBTYPE $OUTPUT/cifar10_test_$DBTYPE

$CAFFEHOME/bin/convert_cifar_data $DATA $OUTPUT $DBTYPE

echo "Computing image mean..."

./cifar10/compute_image_mean -backend=$DBTYPE \
  $OUTPUT/cifar10_train_$DBTYPE $OUTPUT/mean.binaryproto

#./cifar10/compute_image_mean -backend=$DBTYPE \
#  $OUTPUT/cifar10_validate_$DBTYPE $OUTPUT/val_mean.binaryproto
#
#./cifar10/compute_image_mean -backend=$DBTYPE \
#  $OUTPUT/cifar10_test_$DBTYPE $OUTPUT/test_mean.binaryproto
echo "Done."
