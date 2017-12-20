import numpy as np
import tensorflow as tf

def softmax(x):
    """
        Softmax 函数
    """
    assert len(x.shape) > 1, "Softmax的得分向量要求维度高于1"
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x

samples = [
    [1,2,3],
    [4,4,5],
    [3,4,5]
]
samples = np.array(samples)
s = samples.copy()
res = softmax(samples)
print(res)
with tf.variable_scope('foo') as scope:
    s = tf.Variable(s)
a = tf.expand_dims(tf.reduce_max(s, reduction_indices=[1]), 1)
f=tf.reduce_max(s, reduction_indices=[1])

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    sess.run(a)
    print(a.eval())
    print(a.shape)
    print(f.shape)