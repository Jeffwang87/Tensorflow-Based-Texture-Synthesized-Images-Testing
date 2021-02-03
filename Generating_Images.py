import vgg16_avg_pool
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import helper
import tf_helper
import tensorflow as tf
import os

def loss_function(m, texture_op, noise_layers):
    loss = tf.constant(0, dtype=tf.float32, name="Loss")

    for i in range(len(m)):
        texture_filters = np.squeeze(texture_op[m[i][0]], 0)
        texture_filters = np.reshape(texture_filters, newshape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
        gram_matrix_texture = np.matmul(texture_filters.T, texture_filters)

        noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)
        noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
        gram_matrix_noise = tf.matmul(tf.transpose(noise_filters), noise_filters)

        denominator = (4 * tf.convert_to_tensor(texture_filters.shape[1], dtype=tf.float32) * tf.convert_to_tensor(texture_filters.shape[0], dtype=tf.float32))

        loss += m[i][1] * (tf.reduce_sum(tf.square(tf.subtract(gram_matrix_texture, gram_matrix_noise))) / tf.cast(denominator, tf.float32))
    
    return loss

def run_texture_synthesis(input_filename, m, eps, op_dir, initial_filename, final_filename):
    i_w = 256   # width of input image(original image will be scaled down to this width), width of generated image
    i_h = 256   # height of input image(original image will be scaled down to this height), height of generated image
    
    texture_array = helper.resize_and_rescale_img(input_filename, i_w, i_h)
    texture_outputs = tf_helper.compute_tf_output(texture_array)
    
    tf.reset_default_graph()
    vgg = vgg16_avg_pool.Vgg16()

    random_ = tf.random_uniform(shape=texture_array.shape, minval=0, maxval=0.2)
    input_noise = tf.Variable(initial_value=random_, name='input_noise', dtype=tf.float32)

    vgg.build(input_noise)

    noise_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, 6: vgg.conv3_1, 7: vgg.conv3_2,
                   8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4, 14: vgg.conv5_1, 15: vgg.conv5_2,
                   16: vgg.conv5_3, 17: vgg.pool5 })

    loss = loss_function(m, texture_outputs, noise_layers_list)
    optimizer = tf.train.AdamOptimizer().minimize(loss)


    epochs = eps
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        init_noise = sess.run(input_noise)
        for i in range(epochs):
            _, s_loss = sess.run([optimizer, loss])
            if (i+1) % 1000 == 0:
                print("Epoch: {}/{}".format(i+1, epochs), " Loss: ", s_loss)
        final_noise = sess.run(input_noise)
    
    initial_noise = helper.post_process_and_display(init_noise, op_dir, initial_filename, save_file=False)
    final_noise_ = helper.post_process_and_display(final_noise, op_dir, final_filename)
    
m = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1)]
eps = 500
dir = "set your images directory"
for image in sorted(os.listdir(dir)):
    ip_f = dir + '/' + image
    output_dir = "./Output/"
    noise_fn = image + '_noise'
    final_fn = image
    run_texture_synthesis(ip_f, m, eps, output_dir, noise_fn, final_fn)
