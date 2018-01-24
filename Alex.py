from __future__ import print_function
from caffe import layers as L, params as P, to_proto

def conv_bn_relu(bottom, kernel_size, num_output, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                                num_output=num_output, pad=pad, group=group,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant', value=0))
    bn = L.BatchNorm(conv, use_global_stats=False)
    scale = L.Scale(bn, bias_term=True)
    relu = L.ReLU(scale, in_place=True)
    return conv, relu
#global average pooling
def fc_gav(bottom):
    #fc = L.InnerProduct(bottom, num_output=num_output)
    fc = L.Pooling(pool=P.Pooling.AVE, kernel_size=6, stride=1)
    bn = L.BatchNorm(fc, use_global_stats=False)
    scale = L.Scale(bn, bias_term=True)
    relu = L.ReLU(scale, in_place=True)
    drop = L.Dropout(relu, dropout_ratio=0.5)
    return drop

def max_pool(bottom, kernel_size, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride)

def mynet(lmdb, batch_size=256, include_acc=False):
    # data layer
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))
    # the net itself
    #import pdb;pdb.set_trace()
    conv1, relu1 = conv_bn_relu(data, 3, 96, stride=1, pad=2)
    norm1 = L.LRN(relu1, local_size=5, alpha=1e-4, beta=0.75)
    pool1 = max_pool(norm1, 3, stride=2)
    conv2, relu2 = conv_bn_relu(pool1, 3, 128, stride=1, pad=2, group=2)
    norm2 = L.LRN(relu2, local_size=5, alpha=1e-4, beta=0.75)
    pool2 = max_pool(norm2, 3, stride=2)
    conv3, relu3 = conv_bn_relu(pool2, 3, 256, pad=2)
    conv4, relu4 = conv_bn_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = conv_bn_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = L.max_pool(relu5, 3, stride=2)
    fc7 = fc_gav(pool5)
    #fc8 = L.InnerProduct(fc7, num_output=10, kernel_size=1, stride=1,
    #    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant',value=0))
    fc8 = L.Convolution(fc7, kernel_size=1, num_output=10,stride=1,
            weight_filler=dict(type='xavier'),
            bias_filler=dict(type='constant',value=0))
    acc = L.Accuracy(fc8, label)
    loss = L.SoftmaxWithLoss(fc8, label)
    return to_proto(loss, acc)

def make_net():
    #with open('train_val1.prototxt','w') as f:
        #print(mynet('123'), file=f)
    print(mynet('123'))

if __name__ == '__main__':
    make_net()
