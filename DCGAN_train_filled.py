import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from glob import glob
import time
import platform


try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CPU: {platform.processor()}")
if HAS_PSUTIL:
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

print("\n" + "="*60)
print("GPU Detection and Configuration")
print("="*60)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")

gpu_count = torch.cuda.device_count()
print(f"\nPhysical GPU devices found: {gpu_count}")

if gpu_count > 0:
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"    Memory: {gpu_memory:.2f} GB")
    
    device = torch.device('cuda:0')
    test_tensor = torch.randn(3).to(device)
    print(f"GPU Test: Success - tensor created on {test_tensor.device}")
    print("GPU will be used for training")
    USE_GPU = True

else:
    print("Warning: No GPU devices detected")
    device = torch.device('cpu')
    USE_GPU = False

if not USE_GPU:
    print("\nUsing CPU for training (slower)")
else:
    print("\nUsing GPU for training (faster)")

print("="*60 + "\n")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class AirfoilDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((128, 128), dtype=np.uint8)
        
        img = cv2.resize(img, (128, 128))
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.expand_dims(img, axis=0)
        
        if self.transform:
            img = self.transform(img)
        
        return torch.FloatTensor(img)


def preprocess_airfoil_images(input_folder):
    if not os.path.isabs(input_folder):
        input_folder = os.path.join(SCRIPT_DIR, input_folder)
    
    if not os.path.exists(input_folder):
        print(f"Error: Folder not found: {input_folder}")
        return []
    
    image_paths = glob(os.path.join(input_folder, "*.png")) + \
                  glob(os.path.join(input_folder, "*.jpg"))
    print(f"Searching in: {input_folder}")
    print(f"Found {len(image_paths)} images")
    
    return image_paths


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        input = input.view(input.size(0), self.latent_dim, 1, 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),
            nn.Flatten()
        )
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.final(x)
    
    def get_features(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def generate_and_save_images(generator, epoch, latent_dim, output_dir, device, test_input=None):
    generator.eval()
    with torch.no_grad():
        if test_input is None:
            test_input = torch.randn(16, latent_dim).to(device)
        else:
            test_input = test_input.to(device)
        
        predictions = generator(test_input)
        predictions = predictions.cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i, 0, :, :] + 1) / 2.0
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'generated_images_epoch_{:04d}.png'.format(epoch))
    plt.savefig(output_path)
    plt.close()
    generator.train()

def feature_matching_loss(real_features, fake_features):
    real_flat = real_features.view(real_features.size(0), -1)
    fake_flat = fake_features.view(fake_features.size(0), -1)
    return nn.functional.mse_loss(fake_flat, real_flat.detach())

def smoothness_loss(images):
    diff_h = images[:, :, 1:, :] - images[:, :, :-1, :]
    diff_w = images[:, :, :, 1:] - images[:, :, :, :-1]
    smooth_loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)
    return smooth_loss

def train_gan():
    EPOCHS = 300
    BATCH_SIZE = 32
    LATENT_DIM = 100
    G_LR = 2e-4
    D_LR = 2e-4
    INPUT_FOLDER = "Images_airfoil"
    OUTPUT_DIR = "DCGAN_result"
    FEATURE_MATCHING_WEIGHT = 0.1
    SMOOTHNESS_WEIGHT = 0.05
    
    if not os.path.isabs(INPUT_FOLDER):
        INPUT_FOLDER = os.path.join(SCRIPT_DIR, INPUT_FOLDER)
    if not os.path.isabs(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading images from: {INPUT_FOLDER}")
    image_paths = preprocess_airfoil_images(INPUT_FOLDER)
    
    if len(image_paths) == 0:
        print(f"Error: No image files found in {INPUT_FOLDER}")
        return None, None
    
    dataset = AirfoilDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    
    print("Building models...")
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print("\nGenerator Architecture:")
    print(generator)
    print("\nDiscriminator Architecture:")
    print(discriminator)
    
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))
    
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=EPOCHS, eta_min=1e-6)
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=EPOCHS, eta_min=1e-6)

    
    fixed_noise = torch.randn(16, LATENT_DIM).to(device)
    
    print("Starting training...")
    
    gen_losses = []
    disc_losses = []
    
    start_time = time.time()
    
    epoch_range = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch") if HAS_TQDM else range(EPOCHS)
    
    for epoch in epoch_range:
        epoch_start_time = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        num_batches = 0
        
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False, unit="batch") if HAS_TQDM else dataloader
        
        for batch_idx, real_images in enumerate(batch_iter):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)
            
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_images = generator(noise)
            
            discriminator.zero_grad()
            real_output = discriminator(real_images)
            real_loss = criterion(real_output, real_labels)
            
            fake_output_d = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output_d, fake_labels)
            
            disc_loss = (real_loss + fake_loss) / 2.0
            disc_loss.backward()
            discriminator_optimizer.step()
            
            generator.zero_grad()
            fake_output_g = discriminator(fake_images)
            gen_labels = torch.full((batch_size, 1), 1.0, device=device)
            adversarial_loss = criterion(fake_output_g, gen_labels)
            
            real_features = discriminator.get_features(real_images)
            fake_features = discriminator.get_features(fake_images)
            fm_loss = feature_matching_loss(real_features, fake_features)
            
            smooth_loss = smoothness_loss(fake_images)
            
            gen_loss = adversarial_loss + FEATURE_MATCHING_WEIGHT * fm_loss + SMOOTHNESS_WEIGHT * smooth_loss
            gen_loss.backward()
            generator_optimizer.step()
            
            gen_loss_val = gen_loss.item()
            disc_loss_val = disc_loss.item()
            
            total_gen_loss += gen_loss_val
            total_disc_loss += disc_loss_val
            num_batches += 1
            
            if HAS_TQDM:
                batch_iter.set_postfix({
                    'Gen Loss': f'{gen_loss_val:.4f}',
                    'Disc Loss': f'{disc_loss_val:.4f}'
                })
        
        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        elapsed_time = time.time() - start_time
        remaining_epochs = EPOCHS - (epoch + 1)
        eta = (elapsed_time / (epoch + 1)) * remaining_epochs if epoch > 0 else 0
        
        if HAS_TQDM:
            epoch_range.set_postfix({
                'Gen Loss': f'{avg_gen_loss:.4f}',
                'Disc Loss': f'{avg_disc_loss:.4f}',
                'Time': f'{epoch_time:.2f}s',
                'ETA': f'{eta/60:.1f}m'
            })
        
        if epoch % 5 == 0:
            if HAS_TQDM:
                epoch_range.set_description(f"Epoch {epoch+1}/{EPOCHS} - Saving images")
            generate_and_save_images(generator, epoch, LATENT_DIM, OUTPUT_DIR, device, fixed_noise)
            if not HAS_TQDM:
                print(f'Epoch {epoch + 1}/{EPOCHS}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {epoch_time:.2f}s')
        
        if epoch % 20 == 0:
            if HAS_TQDM:
                epoch_range.set_description(f"Epoch {epoch+1}/{EPOCHS} - Saving models")
            torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, f'airfoil_generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, f'airfoil_discriminator_epoch_{epoch}.pth'))
        
        gen_scheduler.step()
        disc_scheduler.step()
    
    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time / 60:.2f} minutes")
    
    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, 'airfoil_generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, 'airfoil_discriminator_final.pth'))
    
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
    plt.close()
    
    return generator, discriminator

def extract_contour_coordinates(pixel_matrix, num_points=200, threshold=127):
    binary_img = (pixel_matrix > threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros((num_points, 2))
    
    main_contour = max(contours, key=cv2.contourArea)
    
    perimeter = cv2.arcLength(main_contour, True)
    epsilon = 0.001 * perimeter
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    contour_points = approx_contour.reshape(-1, 2)
    
    if len(contour_points) >= num_points:
        indices = np.linspace(0, len(contour_points)-1, num_points, dtype=int)
        sampled_points = contour_points[indices]
    else:
        if len(contour_points) > 1:
            t = np.linspace(0, 1, num_points)
            x_coords = np.interp(t, np.linspace(0, 1, len(contour_points)), contour_points[:, 0])
            y_coords = np.interp(t, np.linspace(0, 1, len(contour_points)), contour_points[:, 1])
            sampled_points = np.column_stack((x_coords, y_coords))
        else:
            sampled_points = np.zeros((num_points, 2))
    
    return sampled_points


def save_contour_coordinates(airfoil_images, output_dir):
    coordinates_dir = os.path.join(output_dir, 'airfoil_coordinates')
    os.makedirs(coordinates_dir, exist_ok=True)
    
    for i, img in enumerate(airfoil_images):
        coordinates = extract_contour_coordinates(img, num_points=200)
        
        filename = os.path.join(coordinates_dir, f'airfoil_{i+1:02d}.dat')
        
        with open(filename, 'w') as f:
            for x, y in coordinates:
                f.write(f'{x:.6f} {y:.6f}\n')
    
    print(f'Saved {len(airfoil_images)} airfoil coordinate files to {coordinates_dir}')


def generate_new_airfoils(model_path, output_dir, device, num_samples=9, latent_dim=100):
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        generated_images = generator(noise)
        generated_display = ((generated_images.cpu().numpy() + 1) * 127.5).astype(np.uint8)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(num_samples):
        ax = axes[i // 3, i % 3]
        ax.imshow(generated_display[i, 0, :, :], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Generated Airfoil {i + 1}')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_generated_airfoils.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    pixel_matrices = generated_display[:, 0, :, :]
    save_contour_coordinates(pixel_matrices, output_dir)


def evaluate_generated_airfoils(generator, device, num_samples=20, latent_dim=100):
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        generated_images = generator(noise)
        generated_binary = (generated_images.cpu().numpy() > 0).astype(np.float32)
    
    continuity_scores = []
    
    sample_range = tqdm(range(num_samples), desc="Evaluating") if HAS_TQDM else range(num_samples)
    
    for i in sample_range:
        img = generated_binary[i, 0, :, :]
        
        contour_ratio = np.mean(img)
        
        _, labels = cv2.connectedComponents(img.astype(np.uint8))
        num_components = np.max(labels)
        
        continuity_score = 1.0 / num_components if num_components > 0 else 0
        
        continuity_scores.append(continuity_score)
        
        if not HAS_TQDM:
            print(f"Sample {i + 1}: Contour ratio: {contour_ratio:.4f}, Connected components: {num_components}, Continuity score: {continuity_score:.4f}")
    
    avg_continuity = np.mean(continuity_scores)
    print(f"\nAverage continuity score: {avg_continuity:.4f}")
    
    return avg_continuity


if __name__ == "__main__":
    OUTPUT_DIR = "DCGAN_result"
    if not os.path.isabs(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    
    generator, discriminator = train_gan()
    
    if generator is not None and discriminator is not None:
        model_path = os.path.join(OUTPUT_DIR, 'airfoil_generator_final.pth')
        generate_new_airfoils(model_path, OUTPUT_DIR, device)
        evaluate_generated_airfoils(generator, device)
        print("Training and evaluation completed.")
