import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from glob import glob
import time
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPUs detected; memory growth mode enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

tf.random.set_seed(42)
np.random.seed(42)


def preprocess_airfoil_images(input_folder, output_size=(128, 128)):
    processed_images = []
    image_paths = glob(os.path.join(input_folder, "*.png")) + \
                  glob(os.path.join(input_folder, "*.jpg"))
    print(f"Found {len(image_paths)} images")
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        img = cv2.resize(img, output_size)
        img = (img / 127.5) - 1.0
        img = np.expand_dims(img, axis=-1)
        processed_images.append(img)
    return np.array(processed_images)


def airfoil_generator(latent_dim=100):
    model = keras.Sequential()

    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(8 * 8 * 128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Reshape((8, 8, 128)))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


def airfoil_discriminator():
    model = keras.Sequential()

    model.add(layers.Input(shape=(128, 128, 1)))
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, batch_size, latent_dim, generator, discriminator,
               generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([batch_size, latent_dim])

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

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, latent_dim, test_input=None):
    if test_input is None:
        test_input = tf.random.normal([16, latent_dim])

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i, :, :, 0] + 1) / 2.0
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('generated_images_epoch_{:04d}.png'.format(epoch))
    plt.close()


def train_gan():
    EPOCHS = 200
    BATCH_SIZE = 16
    LATENT_DIM = 100
    LEARNING_RATE = 1e-4

    print("Step 1: Load and preprocess data...")
    images = preprocess_airfoil_images("filled_proportional", output_size=(128, 128))

    if len(images) == 0:
        print("Error: No image files found! Please check the path.")
        return

    print(f"Data set shape: {images.shape}")
    print(f"Pixel value range: [{images.min():.3f}, {images.max():.3f}]")

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=len(images)).batch(BATCH_SIZE)

    print("Step 2: Build the model...")
    generator = airfoil_generator(LATENT_DIM)
    discriminator = airfoil_discriminator()

    print("\nGenerator Architecture:")
    generator.summary()
    print("\nDiscriminator architecture:")
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    fixed_noise = tf.random.normal([16, LATENT_DIM])

    print("Step 3: Begin training...")

    gen_losses = []
    disc_losses = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        num_batches = 0

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(
                image_batch, BATCH_SIZE, LATENT_DIM,
                generator, discriminator,
                generator_optimizer, discriminator_optimizer
            )
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            num_batches += 1

        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        epoch_time = time.time() - epoch_start_time

        if epoch % 20 == 0:
            generate_and_save_images(generator, epoch, LATENT_DIM, fixed_noise)
            print(
                f'Epoch {epoch + 1}/{EPOCHS}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {epoch_time:.2f}s')

        if epoch % 40 == 0:

            generator.save(f'airfoil_generator_epoch_{epoch}.keras')
            discriminator.save(f'airfoil_discriminator_epoch_{epoch}.keras')

    total_time = time.time() - start_time
    print(f"Training complete! Total duration: {total_time / 60:.2f} minutes")

    generator.save('airfoil_generator_final.keras')
    discriminator.save('airfoil_discriminator_final.keras')

    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    return generator, discriminator


def generate_new_airfoils(model_path='airfoil_generator_final.keras', num_samples=9, latent_dim=100):
    generator = tf.keras.models.load_model(model_path)

    noise = tf.random.normal([num_samples, latent_dim])

    generated_images = generator(noise, training=False)

    generated_display = (generated_images + 1) * 127.5
    generated_display = generated_display.numpy().astype(np.uint8)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(num_samples):
        ax = axes[i // 3, i % 3]
        ax.imshow(generated_display[i, :, :, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Generated Airfoil {i + 1}')

    plt.tight_layout()
    plt.savefig('final_generated_airfoils.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_generated_airfoils(generator, num_samples=20, latent_dim=100):
    noise = tf.random.normal([num_samples, latent_dim])
    generated_images = generator(noise, training=False)

    generated_binary = (generated_images > 0).numpy().astype(np.float32)

    continuity_scores = []

    for i in range(num_samples):
        img = generated_binary[i, :, :, 0]

        contour_ratio = np.mean(img)

        _, labels = cv2.connectedComponents(img.astype(np.uint8))
        num_components = np.max(labels)

        continuity_score = 1.0 / num_components if num_components > 0 else 0

        continuity_scores.append(continuity_score)

        print(
            f"Sample {i + 1}: Contour ratio: {contour_ratio:.4f}, Connected components: {num_components}, Continuity score: {continuity_score:.4f}")

    avg_continuity = np.mean(continuity_scores)
    print(f"\nAverage continuity score: {avg_continuity:.4f}")

    return avg_continuity


def image_to_airfoil_coords(image_array, smooth_factor=3, num_points=200):
    img = (image_array + 1) / 2.0
    img = (img * 255).astype(np.uint8)
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    
    if len(contour_points) < 10:
        return None
    
    x_coords = contour_points[:, 0]
    y_coords = contour_points[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    chord_length = x_max - x_min + 1e-8
    
    x_normalized = (x_coords - x_min) / chord_length
    y_normalized = y_coords / chord_length
    
    y_centered = y_normalized - y_normalized.mean()
    
    upper_mask = y_centered >= 0
    lower_mask = y_centered < 0
    
    upper_x = x_normalized[upper_mask]
    upper_y = y_centered[upper_mask]
    lower_x = x_normalized[lower_mask]
    lower_y = y_centered[lower_mask]
    
    if len(upper_x) < 3 or len(lower_x) < 3:
        return None
    
    upper_sorted_idx = np.argsort(upper_x)
    lower_sorted_idx = np.argsort(lower_x)
    
    upper_x_sorted = upper_x[upper_sorted_idx]
    upper_y_sorted = upper_y[upper_sorted_idx]
    lower_x_sorted = lower_x[lower_sorted_idx]
    lower_y_sorted = lower_y[lower_sorted_idx]
    
    upper_x_smooth = gaussian_filter1d(upper_x_sorted, smooth_factor)
    upper_y_smooth = gaussian_filter1d(upper_y_sorted, smooth_factor)
    lower_x_smooth = gaussian_filter1d(lower_x_sorted, smooth_factor)
    lower_y_smooth = gaussian_filter1d(lower_y_sorted, smooth_factor)
    
    x_uniform = np.linspace(0, 1, num_points)
    
    upper_spline = UnivariateSpline(upper_x_smooth, upper_y_smooth, s=len(upper_x_smooth) * 0.01, ext=3)
    lower_spline = UnivariateSpline(lower_x_smooth, lower_y_smooth, s=len(lower_x_smooth) * 0.01, ext=3)
    
    upper_y_interp = upper_spline(x_uniform)
    lower_y_interp = lower_spline(x_uniform)
    
    coords = np.zeros((num_points * 2, 2))
    coords[:num_points, 0] = x_uniform[::-1]
    coords[:num_points, 1] = upper_y_interp[::-1]
    coords[num_points:, 0] = x_uniform
    coords[num_points:, 1] = lower_y_interp
    
    return coords


def save_airfoil_to_xfoil(generator, num_samples=9, latent_dim=100, output_dir='airfoil_coords'):
    os.makedirs(output_dir, exist_ok=True)
    
    noise = tf.random.normal([num_samples, latent_dim])
    generated_images = generator(noise, training=False)
    
    for i in range(num_samples):
        img = generated_images[i, :, :, 0].numpy()
        coords = image_to_airfoil_coords(img)
        
        if coords is not None:
            filename = os.path.join(output_dir, f'airfoil_{i+1:03d}.dat')
            np.savetxt(filename, coords, fmt='%.6f', delimiter='\t')


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Using the GPU: {physical_devices[0]}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Using CPU")

    generator, discriminator = train_gan()

    print("Generate new airfoil...")
    generate_new_airfoils()

    print("Evaluating generated airfoils...")
    evaluate_generated_airfoils(generator)

    print("Converting generated images to airfoil coordinates...")
    save_airfoil_to_xfoil(generator)

    print("All done!")