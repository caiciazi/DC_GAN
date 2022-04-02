import tensorflow as tf
from tensorflow.keras import layers

class make_generator_model(tf.keras.Model):
    def __init__(self):
        super(make_generator_model, self).__init__()
        self.fc = layers.Dense(7*7*256, input_shape=(100,))
        self.bn0 = layers.BatchNormalization()
        self.reshape = layers.Reshape((7, 7, 256))

        self.conv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')


    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = self.bn0(x, training)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)

        x = self.conv1(x)
        x = self.bn1(x, training)
        x = tf.nn.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training)
        x = tf.nn.leaky_relu(x)

        out = self.conv3(x)
        return out

class make_discriminator_model(tf.keras.Model):
    def __init__(self):
        super(make_discriminator_model, self).__init__()
        self.conv0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 3])
        self.LR0 = layers.LeakyReLU()
        self.drop0 = layers.Dropout(0.3)

        self.conv1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.LR1 = layers.LeakyReLU()
        self.drop1 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0(inputs)
        x = self.LR0(x)
        x = self.drop0(x)

        x = self.conv1(x)
        x = self.LR1(x)
        x = self.drop1(x)

        x = self.flatten(x)
        out = self.dense(x)
        return out


if __name__ == '__main__':
    noise = tf.random.normal((2, 100), dtype=tf.float32)
    G = make_generator_model()
    result = G(noise)
    D = make_discriminator_model()
    result1 = D(result)
    print(result1)




    