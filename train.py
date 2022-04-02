from model.model_class import make_generator_model, make_discriminator_model
import tensorflow as tf
import dataset


train_dataset = dataset.main()
train_dataset = train_dataset.shuffle(20000).batch(2048, drop_remainder=True)
print(train_dataset)
generator = make_generator_model()
discriminator = make_discriminator_model()

# losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# we hope the discriminator can make real_loss'result close to 1,and make fake_loss'result close to 0
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
# we hope generator can make fake_output'result close to 1
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)

# # checkpoint
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)




def train(ds, epochs):
    for epoch in range(epochs):
        for step, images in enumerate(ds):
            noise = tf.random.normal([2048, 100])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            print('epoch:{} step:{} gen_loss:{} disc_loss:{}'.format(epoch, step, float(gen_loss), float(disc_loss)))


train(train_dataset, epochs=50)