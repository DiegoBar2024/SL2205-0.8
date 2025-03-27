import tensorflow as tf

## Especifico ruta de la imagen

# Read images (of size 255 x 255) from file.
im1 = tf.image.decode_image(tf.io.read_file('imagen1.png'))
im2 = tf.image.decode_image(tf.io.read_file('imagen1.png'))
tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
# Add an outer batch for each image.
im1 = tf.expand_dims(im1, axis=0)
im2 = tf.expand_dims(im2, axis=0)
# Compute SSIM over tf.uint8 Tensors.
ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)

# Compute SSIM over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)
# ssim1 and ssim2 both have type tf.float32 and are almost equal