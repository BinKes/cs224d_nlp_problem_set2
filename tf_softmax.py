'''
Created on 2017年9月21日

@author: weizhen
'''
import tensorflow as tf
import numpy as np
def softmax(x):
    """
    tensorflow 版本的softmax函数
    compute the softmax function in tensorflow
    interal functions may be used:
    tf.exp,tf.reduce_max,tf.reduce_sum,tf.expend_dims
    Args:
        x:tf.Tensor with shape (n_samples,n_features)
        feature vectors are represented by row-vectors (no need to handle 1-d
        input as in the previous homework)
    Returns:
        out:tf.Tensor with shape (n_sample,n_features). You need to construct
        this tensor in this problem
    """
    
    # tf.reduce_max沿着tensorflow的某一维度计算元素的最大值
    # tf.reduce_sum沿着tensorflow的某一维度计算元素的和
    # tf.expand_dims在tensorflow的某一维度插入一个tensor

    # 得到每一行样本的最大值，得到一个最大值行向量，沿着idx=1（第二个）维度添加一个纬度
    # 如 [1,2,3] shape = (3,), idx=1的维度上加一个维度得到[[1],[2], [3]] shape=(3,1)
    maxes = tf.expand_dims(tf.reduce_max(x, reduction_indices=[1]), 1)
    x_red = x - maxes # 标准化，softmax计算得到的概率不变。数值稳定性
    # 减最大值问题：
    '''
    解释1：
    当我们运算比较小的值的时候是不会有什么问题的，但是如果运算的值比较大的时候，比如 很大或很小的时候，朴素的直接计算会上溢出或下溢出，从而导致严重问题。

    举个例子，对于[3,1,-3]，直接计算是可行的，我们可以得到(0.88,0.12,0)。

    但对于[1000,1000,1000]，却并不可行，我们会得到inf(这也是深度学习训练过程常见的一个错误，看了本文之后，以后出现inf的时候，至少可以考虑softmax运算的上溢和下溢)；对于[-1000,-999,-1000]，还是不行，我们会得到-inf。

    这是因为你的浮点数只有64位，在计算指数函数的环节，exp{1000} =inf，会发生上溢出；exp{-1000} =0，会发生下溢出。
    解释2：
    # 为什么要减去最大值？数值稳定性的解释。#

    在求 exponential 之前将x 的每一个元素减去x_i 的最大值。这样求 exponential 的时候会碰到的最大的数就是 0 了，不会发生overflow 的问题，但是如果其他数原本是正常范围，现在全部被减去了一个非常大的数，于是都变成了绝对值非常大的负数，所以全部都会发生 underflow，但是underflow 的时候得到的是 0，这其实是非常 meaningful 的近似值，而且后续的计算也不会出现奇怪的 NaN。

    当然，总不能在计算的时候平白无故地减去一个什么数，但是在这个情况里是可以这么做的，因为最后的结果要做 normalization，很容易可以证明，这里对x 的所有元素同时减去一个任意数都是不会改变最终结果的——当然这只是形式上，或者说“数学上”，但是数值上我们已经看到了，会有很大的差别。
    http://blog.csdn.net/mzpmzk/article/details/53083579
    '''
    x_exp = tf.exp(x_red)
    sums = tf.expand_dims(tf.reduce_sum(x_exp, reduction_indices=[1]), 1)
    out = x_exp / sums
    
    return out

def cross_entropy_loss(y, yhat):
    """
                  计算交叉熵在tensorflow中
       y是一个one-hot tensor  大小是(n_samples,n_classes)这么大，类型是tf.int32
       yhat是一个tensor 大小是(n_samples,n_classes)  类型是 tf.float32
       function:
           tf.to_float,tf.reduce_sum,tf.log可能会用到
                  参数:
           y:tf.Tensor with shape(n_samples,n_classes) One-hot encoded
           yhat: tf.Tensorwith shape (n_samples,n_classes) Each row encodes a
                probability distribution and should sum to 1
                  返回:
           out: tf.Tensor with shape(1,) (Scalar output).You need to construct
              this tensor in the problem.
    """
    y = tf.to_float(y)
    out = -tf.reduce_sum(y * tf.log(yhat))
    return out

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive
    """
    print("Running basic tests...")
    test1 = softmax(tf.convert_to_tensor(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6
    test2 = softmax(tf.convert_to_tensor(np.array([[-1001, -1002]]), dtype=tf.float32))
    # np.fabs 计算非复数绝对值
    with tf.Session():
        test2 = test2.eval()
    assert np.amax(np.fabs(test2 - np.array([0.73105858, 0.26894142]))) <= 1e-6
    print("Basic (non-exhaustive) softmax tests pass\n")
    
def test_cross_entropy_loss_basic():
    """
    Some simple tests to get you started
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])
    
    test1 = cross_entropy_loss(tf.convert_to_tensor(y, dtype=tf.int32),
                               tf.convert_to_tensor(yhat, dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    result = -3 * np.log(.5)
    assert np.amax(np.fabs(test1 - result)) <= 1e-6
    print("Basic (non-exhaustive) cross-entropy tests pass\n")
    
if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
