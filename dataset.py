import tensorflow as tf
import pathlib
import random
def get_path():
    data_root_orig = r'C:\Users\Administrator\Desktop\GAN_cartoon'
    data_root = pathlib.Path(data_root_orig)



    all_image_paths = list(data_root.glob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    return all_image_paths

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [28, 28])
  image = (image / 255.0) * 2 - 1  # normalize to [-1,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)# img_raw
  return preprocess_image(image)


def bulid_ds(path):
    path_ds = tf.data.Dataset.from_tensor_slices(path)
    ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)



    return ds
def main():
    all_image_paths = get_path()
    ds = bulid_ds(all_image_paths)
    return ds
if __name__ == '__main__':
    ds = main()
    print(ds)

