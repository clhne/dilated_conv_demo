import tensorflow as tf

img = tf.constant(value=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=tf.float32)
img = tf.expand_dims(img, 0)
img = tf.expand_dims(img, -1)
img2 = tf.concat(values=[img,img],axis=3)

filter = tf.constant(value=1, shape=[3,3,2,5], dtype=tf.float32)

out_img1 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=1, padding='SAME')
out_img11 = tf.nn.conv2d(input=img2, filter=filter, strides=[1,1,1,1], padding='SAME')
out_img2 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=1, padding='VALID')
out_img3 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=2, padding='SAME')

#error
#out_img4 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='VALID')

with tf.Session() as sess:
    print('rate=1, SAME mode result:')
    print(sess.run(out_img1))
    print('strides=1, SAME mode result:')
    print(sess.run(out_img11))
    print('rate=1, VALID mode result:')
    print(sess.run(out_img2))
    print('rate=2, SAME mode result:')
    print(sess.run(out_img3))
    print(sess.run(img))
    # error
    #print 'rate=2, VALID mode result:'
    #print(sess.run(out_img4))