import tensorflow as tf

def transform(feature_tensor, theta_tensor, out_height=40, out_width=40):
    # Form as following :
    # [ [t1, t2, t3]
    #   [t4, t5, t6] ]
    batch_num, img_height, img_width, img_channel = feature_tensor.shape    
    transformation_matrix = tf.reshape(theta_tensor, [-1, 2, 3])

    # Generate mesh grid points
    grids = generateMeshGrid(out_height, out_width)

    # Transform process
    #T_g = tf.scan(lambda prev, curr: tf.matmul(curr, grids), transformation_matrix, infer_shape=False)
    transformation_matrix = tf.reshape(transformation_matrix, [-1, 3])
    T_g = tf.matmul(transformation_matrix, grids)
    T_g = tf.reshape(T_g, [-1, 2, 1600])
    
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = tf.reshape(x_s, [1, -1])
    y_s_flat = tf.reshape(y_s, [1, -1])

    # Interpolate the V intensity
    transformed_tensor = interpolate(feature_tensor, x_s_flat, y_s_flat, out_height, out_width)
    transformed_tensor = tf.reshape(transformed_tensor, [None])

def generateMeshGrid(out_height, out_width):
    x_t, y_t = tf.meshgrid(tf.lin_space(-1.0, 1.0, out_width),
        tf.lin_space(-1.0, 1.0, out_height)
    )
    x_t_flat = tf.reshape(x_t, [1, -1])
    y_t_flat = tf.reshape(y_t, [1, -1])
    ones = tf.ones_like(x_t_flat)
    grids = tf.concat((x_t_flat, y_t_flat, ones), axis=0)
    print(grids.shape)
    return grids

def interpolate(imgs_tensor, x, y, out_height, out_width):
    batch_num, img_height, img_width, img_channel = imgs_tensor.shape
    img_height = tf.cast(img_height, tf.float32)
    img_width = tf.cast(img_width, tf.float32)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (img_width - 1)
    y = (y + 1) / 2 * (img_height - 1)

    # Obtain indices
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = tf.cast(x0, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    x1 = tf.cast(tf.minimum(x1, img_width - 1), tf.int32)
    y1 = tf.cast(tf.minimum(y1, img_height - 1), tf.int32)

    # Flat the indices (???)
    single_pixel_indice = tf.range(batch_num, dtype=tf.int32) * imgs_tensor.shape[1] * imgs_tensor.shape[2]
    base = tf.tile(single_pixel_indice, [out_height * out_width])


    #base = tf.reshape(base, [-1, img_height * img_width])
    base_y0 = base + y0 * tf.cast(img_width, tf.int32)
    base_y1 = base + y1 * tf.cast(img_width, tf.int32)
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Look up pixel
    print('list type: ', type(imgs_tensor))
    imgs_flat = tf.reshape(imgs_tensor, [-1, 1])
    Ia = imgs_flat[idx_a]
    Ib = imgs_flat[idx_b]
    Ic = imgs_flat[idx_c]
    Id = imgs_flat[idx_d]

    # Calculate the interpolated values
    wa = tf.expand_dims((x1 - x) * (y1 - y), axis=-1)
    wb = tf.expand_dims((x1 - x) * (y - y0), axis=-1)
    wc = tf.expand_dims((x - x0) * (y1 - y), axis=-1)
    wd = tf.expand_dims((x - x0) * (y - y0), axis=-1)
    output = tf.reduce_sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    return output


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, [32, 100, 100, 1])
    theta = tf.placeholder(tf.float32, [32, 6])
    transform(imgs, theta)