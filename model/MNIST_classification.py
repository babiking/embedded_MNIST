import tensorflow as tf
from utils import utils
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os






class mnist_classification(object):

    def __init__(self, sess, graph, train_param={'num_of_epoches': 5,
                                        'num_of_classes': 10,
                                        'log_dir': './log',
                                        'checkpoint_dir': './checkpoint',
                                        'checkpoint_name': 'mnist_train',
                                        'batch_size': 128,
                                        'learn_rate': 1e-4,
                                        'max_iter': 5000,
                                        'dim_feat': 28}):

        self.num_of_epoches = train_param['num_of_epoches']
        self.log_dir = train_param['log_dir']

        self.checkpoint_dir    = train_param['checkpoint_dir']
        self.checkpoint_name   = train_param['checkpoint_name']

        self.batch_size   = train_param['batch_size']

        self.learn_rate = train_param['learn_rate']
        self.max_iter   = train_param['max_iter']

        self.dim_feat = train_param['dim_feat']
        self.num_of_classes = train_param['num_of_classes']

        self.sess = sess
        self.graph = graph

        # !Build-up MNIST classification model...
        assert self.sess.graph is self.graph
        self._build_model()

        self.saver = tf.train.Saver()




    def _convolution_block(self, inp_feat, kernel_size, num_of_kernel_channels, conv_strides, conv_padding, var_scope):
        '''
            Function:
                        _convolution_block, i.e. convolution + Maxpooling + ReLU
            Input:
                    [1] <tensor> inp_feat, i.e. input feature, dimension->[batch_size, height, width, channel]
                    [2] <int32>  kernel_size
                    [3] <int32> num_of_kernel_channels
                    [4] <int32> conv_strides
                    [5] <string> conv_padding
                    [5] <string> var_scope
            Output:
                    <tensor> activ
        '''
        try:
            num_of_feat_channels = inp_feat.shape[3].value
        except:
            num_of_feat_channels = 1

        with tf.variable_scope(var_scope):
            weights = tf.get_variable(name='conv_weights', shape=[kernel_size, kernel_size, num_of_feat_channels, num_of_kernel_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

            biases = tf.get_variable(name='conv_biases', shape=[num_of_kernel_channels], initializer=tf.zeros_initializer())

        # !Convolution layer...
        conv = tf.nn.conv2d(inp_feat, weights, strides=conv_strides, padding=conv_padding, name='conv', data_format='NHWC') + biases

        # !Activation layer...
        activ = tf.nn.relu(conv)

        # !Pooling layer...
        pool = tf.nn.max_pool(activ,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        return pool





    def _fc_layer(self, inp_feat, num_of_outputs, var_scope):
        '''
            Function:
                        _fc_layer
            Input:
                        [1] inp_feat, dimension->[batch_size, dim_feat]
                        [2] num_of_outputs
                        [3] var_scope: reuse=True
            Output:
                        fc
        '''

        dim_feat = inp_feat.shape[1].value

        with tf.variable_scope(var_scope):

            fc_weights = tf.get_variable(name='fc_weights', dtype=tf.float32, shape=[dim_feat, num_of_outputs], initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc_bias    = tf.get_variable(name='fc_bias',    dtype=tf.float32, shape=[num_of_outputs], initializer=tf.zeros_initializer())


            # tf.nn.xw_plus_b(x, weights, bias) = tf.matmul(x, weights) + biases
            fc = tf.nn.xw_plus_b(x=inp_feat, weights=fc_weights, biases=fc_bias)

            return fc








    def _build_model(self):

        self.digit = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_feat, self.dim_feat, 1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_classes])

        # # !One-hot encoding for label...
        # with tf.name_scope('label_trans'):
        #     self.new_label = utils._array_sparse_to_dense(self.label, self.num_of_classes)

        conv1 = self._convolution_block(inp_feat=self.digit, conv_strides=[1,1,1,1], conv_padding='SAME', kernel_size=5, num_of_kernel_channels=32, var_scope='conv1')

        conv2 = self._convolution_block(inp_feat=conv1, conv_strides=[1,1,1,1], conv_padding='SAME', kernel_size=5, num_of_kernel_channels=64, var_scope='conv2')

        conv3 = self._convolution_block(inp_feat=conv2, conv_strides=[1, 1, 1, 1], conv_padding='SAME', kernel_size=7,num_of_kernel_channels=64, var_scope='conv3')

        flatt = tf.layers.flatten(conv3, name='flatten')

        fc1 = self._fc_layer(inp_feat=flatt, num_of_outputs=1024, var_scope='fc1')
        fc1 = tf.nn.relu(fc1)

        # !Dropout...
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

        fc2 = self._fc_layer(inp_feat=fc1, num_of_outputs=10, var_scope='fc2')

        self.predict = tf.nn.softmax(logits=fc2, dim=-1)

        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=fc2))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.predict), reduction_indices=[1]))

        # !Define MNIST classification accuracy...
        correct = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # !Define training operations...
        self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)





    def _train(self, img_file, lab_file):

        # !Load data into memory...
        img, n_rows, n_cols = utils._read_MNIST_file(img_file, fmt='>IIII')
        lab, _, _ = utils._read_MNIST_file(lab_file, fmt='>II')


        # !Intialize the global_variables and local_variables...
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # # !List all trainable variables...
        train_vars = tf.trainable_variables()
        # for var in train_vars:
        #     print var.name


        # !Add gradients into tensorboard summary...
        gradients = tf.gradients(self.loss, train_vars)
        for (grad, var) in enumerate(gradients):
            tf.summary.histogram(var.name, grad)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#



        # !Add loss into tensorboard summary...
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

        tf.summary.image('input_image', self.digit)

        summary_writer = tf.summary.FileWriter("./log", self.sess.graph)

        merged = tf.summary.merge_all()

        # !Start training process...
        for ii_epoch in range(self.num_of_epoches):
            for ii_iter in range(self.max_iter):

                img_batch, indx = utils._randomly_sample(img, self.batch_size)
                img_batch = np.expand_dims(img_batch, axis=3) / 255

                lab_batch = lab[indx]
                lab_batch = utils._array_sparse_to_dense(np.int64(lab_batch), num_of_classes=self.num_of_classes)


                _, pred, los, acc, summary= self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, merged], feed_dict={self.digit: img_batch, self.label: lab_batch})

                if ( (ii_epoch * self.max_iter + ii_iter) % 1000 == 0):
                    summary_writer.add_summary(summary, ii_epoch * self.max_iter + ii_iter)
                    self._save_model(ii_epoch * self.max_iter + ii_iter)
                    print('!Loss at No.%d Epoch, No.%d Iteration=%.5f' % (ii_epoch, ii_iter, los))
                    print('!Accuracy at No.%d Epoch, No.%d Iteration=%.5f' % (ii_epoch, ii_iter, acc))


    def _test(self, img_test, load_step=-1):
        '''
            Function:
                        _test_phase
        '''
        if self._load_model(load_step):
            predict_test = self.sess.run(self.predict, feed_dict={self.digit: img_test})
            return predict_test
        else:
            return None


    def _save_model(self, iter_step):
        '''
            Function:
                        _save_model, i.e. save trained models during each iteration time step...
        '''

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.checkpoint_name),
                        global_step=iter_step)


    def _load_model(self, load_step):
        '''
            Function:
                        _load_model
        '''
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = (self.checkpoint_name + '-%d') % load_step

            if not os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name+'.meta')):
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))

            return True
        else:
            return False


