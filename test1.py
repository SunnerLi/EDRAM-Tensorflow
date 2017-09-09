import numpy as np
import theano.tensor as T


batch_size=2
num_channel = 1
height = 2
width = 2

origin = np.reshape(range(batch_size * num_channel * height * width), [batch_size, num_channel, height, width])
print(origin)

origin_batch = np.asarray(
    [
        [
            [[0], [1]],
            [[2], [3]]
        ],
        [
            [[4], [5]],
            [[6], [7]]
        ]
    ]
)
print(np.shape(np.swapaxes(np.swapaxes(origin, 1, 2), 2, 3)))
print(np.swapaxes(np.swapaxes(origin, 1, 2), 2, 3))


base = T.repeat(T.arange(32, dtype='int64') * 10000, 40 * 40)
print(base.eval())