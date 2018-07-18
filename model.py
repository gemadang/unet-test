from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from skimage import io

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Input, Cropping2D
from keras.layers import BatchNormalization, Concatenate, Activation, Dropout, Reshape
from keras.initializers import RandomNormal

import numpy as np
import itertools
import os

import time

EPOCHS=1000
BATCH_SIZE=35
SAVE_INTERVAL=50

def one_hot_it(labels, shape=(360,480,12)):
    x = np.zeros(shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j,int(labels[i][j])]=1
    return x

class SegNet():
    def __init__(self):
        self.image_rows = 256
        self.image_cols = 256
        self.channels = 3
        self.labels = 2
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        self.label_shape = (self.image_rows, self.image_cols, self.labels)
        self.data_shape = self.image_cols * self.image_rows

        self.dataset_name = '/storage'

        self.label_colours = np.array([[0, 0, 0], [255, 255, 255]])

        self.conv_init = RandomNormal(0, 0.02)
        self.gamma_init = RandomNormal(1., 0.02)

        self.model = self.build_model()

        self.model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    def build_model(self):
        in_img = Input(self.image_shape, name="model_in")

        d0 = Conv2D(64, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(in_img)
        d0 = LeakyReLU(alpha=0.2)(d0)

        d1 = Conv2D(128, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d0)
        d1 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d1, training=1)
        d1 = LeakyReLU(alpha=0.2)(d1)

        d2 = Conv2D(256, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d1)
        d2 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d2, training=1)
        d2 = LeakyReLU(alpha=0.2)(d2)

        d3 = Conv2D(512, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d2)
        d3 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d3, training=1)
        d3 = LeakyReLU(alpha=0.2)(d3)

        d4 = Conv2D(512, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d3)
        d4 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d4, training=1)
        d4 = LeakyReLU(alpha=0.2)(d4)

        d5 = Conv2D(512, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d4)
        d5 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d5, training=1)
        d5 = LeakyReLU(alpha=0.2)(d5)

        d6 = Conv2D(512, kernel_initializer=self.conv_init, kernel_size=3, strides=2, padding='same')(d5)
        d6 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d6, training=1)
        d6 = LeakyReLU(alpha=0.2)(d6)

        d7 = Conv2D(512, kernel_initializer=self.conv_init, kernel_size=4, strides=2, padding='same', use_bias=True)(d6)
        d7 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(d7, training=1)
        d7 = Activation("relu")(d7)

        u0 = Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(d7)
        u0 = Cropping2D(1)(u0)
        u0 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u0, training=1)
        u0 = Dropout(0.5)(u0, training=1)
        u0 = Concatenate()([u0, d6])
        u0 = Activation("relu")(u0)

        u1 = Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u0)
        u1 = Cropping2D(1)(u1)
        u1 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u1, training=1)
        u1 = Dropout(0.5)(u1, training=1)
        u1 = Concatenate()([u1, d5])
        u1 = Activation("relu")(u1)

        u2 = Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u1)
        u2 = Cropping2D(1)(u2)
        u2 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u2, training=1)
        u2 = Dropout(0.5)(u2, training=1)
        u2 = Concatenate()([u2, d4])
        u2 = Activation("relu")(u2)

        u3 = Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u2)
        u3 = Cropping2D(1)(u3)
        u3 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u3, training=1)
        u3 = Dropout(0.5)(u3, training=1)
        u3 = Concatenate()([u3, d3])
        u3 = Activation("relu")(u3)

        u4 = Conv2DTranspose(256, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u3)
        u4 = Cropping2D(1)(u4)
        u4 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u4, training=1)
        u4 = Dropout(0.5)(u4, training=1)
        u4 = Concatenate()([u4, d2])
        u4 = Activation("relu")(u4)

        u5 = Conv2DTranspose(128, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u4)
        u5 = Cropping2D(1)(u5)
        u5 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u5, training=1)
        u5 = Concatenate()([u5, d1])
        u5 = Activation("relu")(u5)

        u6 = Conv2DTranspose(64, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u5)
        u6 = Cropping2D(1)(u6)
        u6 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u6, training=1)
        u6 = Concatenate()([u6, d0])
        u6 = Activation("relu")(u6)

        u7 = Conv2DTranspose(self.labels, kernel_size=4, strides=2, kernel_initializer=self.conv_init)(u6)
        u7 = Cropping2D(1)(u7)
        u7 = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                gamma_initializer=self.gamma_init)(u7, training=1)
        out_img = Activation('softmax')(u7)

        model = Model(in_img, out_img)
        model.summary()

        return model

    def train(self, epochs, batch_size=128, save_interval=50):
        for epoch in range(epochs):
            train_data, train_labels = self.load_data(batch_size)

            loss = self.model.train_on_batch(train_data, train_labels)
            print("%d [loss1: %f loss2: %f]" % (epoch, loss[0], loss[1]))

            if epoch % save_interval == 0:
                self.test_model(epoch)
                self.save("SegNet_epoch_%d" % (epoch))

        self.test_model(epochs)
        self.save("SegNet_final")

    def load_data(self, batch_size=128):
        train_path = glob('./%s/Train_data/*' % (self.dataset_name))
        #train_path = glob('./data/train/*')

        batch_images = np.random.choice(train_path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img_name = img_path.split('/')[-1]

            img = io.imread(img_path)
            label = one_hot_it(io.imread('./%s/Train_label/%s' % (self.dataset_name, img_name)), shape=self.label_shape)
            #label = one_hot_it(io.imread('./data/train_labels/%s' % (img_name)), shape=self.label_shape)

            if np.random.random() < 0.5:
                img = np.fliplr(img)
                label = np.fliplr(label)

            imgs_A.append(img)
            imgs_B.append(label)

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B)
        return imgs_A, imgs_B

    def test_model(self, epoch):
        r, c = 3, 3

        test_data, test_labels = self.load_data(batch_size=3)

        segment = self.model.predict(test_data)

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0, i].imshow(test_data[i, :, :, :])
            axs[0, i].axis('off')
            axs[1, i].imshow(self.visualize(np.argmax(test_labels[i, :, :, :], axis=2)))
            axs[1, i].axis('off')
            filled_in = self.visualize(np.argmax(segment[i, :, :, :], axis=2))
            axs[2, i].imshow(filled_in)
            axs[2, i].axis('off')
        #fig.savefig("/artifacts/output/img_results/training_%d.png" % (epoch))
        fig.savefig("/artifacts/training_%d.png" % (epoch))
        plt.close()

    def visualize(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.labels):
            r[temp == l] = self.label_colours[l, 0]
            g[temp == l] = self.label_colours[l, 1]
            b[temp == l] = self.label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r / 255.0)  # [:,:,0]
        rgb[:, :, 1] = (g / 255.0)  # [:,:,1]
        rgb[:, :, 2] = (b / 255.0)  # [:,:,2]

        return rgb

    def save(self, model_name):
        '''
        model_path = "/artifacts/output/saved_model/%s.json" % (model_name)
        weights_path = "/artifacts/output/saved_model/%s_weights.hdf5" % (model_name)
        '''
        model_path = "/artifacts/%s.json" % (model_name)
        weights_path = "/artifacts/%s_weights.hdf5" % (model_name)
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = self.model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        self.model.save_weights(options['file_weight'])


if __name__ == '__main__':
    time_start = time.time()

    segnet = SegNet()
    segnet.train(epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=SAVE_INTERVAL)
    segnet.save('final')

    time_elapsed = (time.time() - time_start)
    with open("/artifacts/time_metrics.txt", "w") as text_file:
        text_file.write("Time Runnning: {} Epochs: {}".format(time_elapsed, EPOCHS))
