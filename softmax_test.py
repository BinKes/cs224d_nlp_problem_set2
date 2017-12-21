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
res = softmax(s)
print(res)
ids = np.random.permutation(len(samples))
print('samples:',samples[ids])

with tf.Graph().as_default() as g:
    with tf.variable_scope('foo') as scope:
        s = tf.Variable(s)
        a = tf.expand_dims(tf.reduce_max(s, reduction_indices=[1]), 1, name='expand_test')
    f=tf.reduce_max(s, reduction_indices=[1])

    with tf.Session() as sess:
        with tf.variable_scope('foo') as scope:
            r=g.get_tensor_by_name(name='foo/expand_test:0')
        
        init = tf.global_variables_initializer()
        sess.run(init)
        c = sess.run([a,r])
        print(a.eval())