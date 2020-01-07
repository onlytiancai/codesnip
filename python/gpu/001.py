import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with tf.device('/cpu:0'):
    v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
    v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
    sumV12 = v1 + v2

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(sumV12))
