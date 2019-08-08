import tensorflow as tf

class NN():
    def __init__(self, dense_layers, input_dim, output_dim, batch_size, act, learning_rate, keep_rate):
        self.dense_layers = dense_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.act = act
        self.learning_rate = learning_rate
        self.keep_rate = keep_rate

        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim])
        self.y = tf.placeholder(dtype=tf.int32, shape=[batch_size, output_dim])
        self.model = tf.placeholder(dtype=tf.string)


        self.model()

    def cnn_weight_init(self):
        weight1 = tf.Variable(tf.truncated_normal(shape=[5, 1, 200], stddev=0.1))
        weight2 = tf.Variable(tf.truncated_normal(shape=[4, 200, 100], stddev=0.1))
        return weight1, weight2

    def dense_weight_init(self, num_cells, input_dim):
        weight = tf.Variable(tf.truncated_normal(shape=[num_cells, input_dim], stddev=0.1, dtype=tf.float32))
        return weight

    def dense_bias_init(self, num_cells):
        bias = tf.Variable(tf.zeros(shape=[num_cells, 1], dtype=tf.float32))
        return bias

#没有激活层，论文里没写
    def create_cnn(self):
        weight1, weight2 = self.cnn_weight_init()
        conv1 = tf.nn.conv1d(self.x, weight1, stride=2, padding='VALID')
        if self.model == 'train':
            conv1 = tf.nn.dropout(conv1, keep_prob=0.95)
        act = tf.nn.relu(conv1)
        conv2 = tf.nn.conv1d(act, weight2, stride=1, padding='VALID')
        act = tf.nn.relu(conv2)
        #batch, window_size/stride, channel
        avg_pool = tf.nn.avg_pool(act, ksize=[1, 2, 1], strides=[1, 2, 1], padding='VALID')
        #avg_pool = tf.nn.pool(conv2, window_shape=[2], pooling_type='AVG', padding='VALID', strides=[2])
        output_shape = tf.shape(avg_pool)
        x_data = tf.reshape(avg_pool, shape=[output_shape[0]*output_shape[1], 1])
        if self.model == 'train':
            x_data = tf.nn.dropout(x_data, keep_prob=0.75)
        return x_data

    def create_dense(self, x_data):
        for num_cells in self.dense_layers:
            input_dim = tf.shape(x_data)[0]
            weight = self.dense_weight_init(num_cells, input_dim)
            bias = self.dense_bias_init(num_cells)
            mat = tf.matmul(weight, x_data)
            add = tf.add(mat, bias)
            x_data = self.act(add)
        output = tf.nn.softmax(x_data)
        return output

    def calculate_loss(self, output):
        loss = tf.reduce_mean((output - self.y) ** 2)
        return loss

    def cal_accuracy(self, output):
        result = tf.argmax(output, axis=1)
        label = tf.argmax(self.y, axis=1)
        bool_compare = tf.equal(result, label)
        float_compare = tf.cast(bool_compare, tf.float32)
        accuracy = tf.reduce_mean(float_compare)
        return accuracy

    def model(self):
        x_data = self.create_cnn()
        output = self.create_dense(x_data)
        if self.model == 'test':
            self.accuracy = self.cal_accuracy(output)
        self.loss = loss = self.calculate_loss(output)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

