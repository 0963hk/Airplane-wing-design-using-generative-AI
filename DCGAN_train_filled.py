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
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

def extract_contour_from_image(binary_img):
    binary_uint8 = (binary_img * 255).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 10:
        return None
    
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(largest_contour) < 10:
        return None
    
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    
    h, w = binary_img.shape
    contour_points[:, 0] = contour_points[:, 0] / w
    contour_points[:, 1] = (h - contour_points[:, 1]) / h
    
    return contour_points

def gaussian_smooth_1d(data, sigma=2.0):
    if sigma <= 0:
        return data
    kernel_size = int(6 * sigma) | 1
    kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(data, kernel_size // 2, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

def resample_contour_to_points(contour_points, num_points=200):
    if contour_points is None or len(contour_points) < 3:
        return None
    
    points = contour_points.copy()
    
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    total_distance = distances[-1]
    if total_distance < 1e-6:
        return None
    
    distances = distances / total_distance
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    t_new = np.linspace(0, 1, num_points)
    
    if HAS_SCIPY:
        try:
            tck_x, u_x = interpolate.splprep([x_coords], s=0, per=True, k=min(3, len(points)-1))
            tck_y, u_y = interpolate.splprep([y_coords], s=0, per=True, k=min(3, len(points)-1))
            
            x_smooth, _ = interpolate.splev(t_new, tck_x)
            y_smooth, _ = interpolate.splev(t_new, tck_y)
            
            resampled = np.column_stack([x_smooth, y_smooth])
        except:
            f_x = interpolate.interp1d(distances, x_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')
            f_y = interpolate.interp1d(distances, y_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            x_resampled = f_x(t_new)
            y_resampled = f_y(t_new)
            
            x_resampled = gaussian_filter1d(x_resampled, sigma=2.0)
            y_resampled = gaussian_filter1d(y_resampled, sigma=2.0)
            
            resampled = np.column_stack([x_resampled, y_resampled])
    else:
        x_resampled = np.interp(t_new, distances, x_coords)
        y_resampled = np.interp(t_new, distances, y_coords)
        
        x_resampled = gaussian_smooth_1d(x_resampled, sigma=2.0)
        y_resampled = gaussian_smooth_1d(y_resampled, sigma=2.0)
        
        resampled = np.column_stack([x_resampled, y_resampled])
    
    resampled[:, 0] = np.clip(resampled[:, 0], 0, 1)
    resampled[:, 1] = np.clip(resampled[:, 1], 0, 1)
    
    return resampled

def normalize_airfoil_coordinates(coords):
    if coords is None or len(coords) < 10:
        return None
    
    coords = coords.copy()
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
        return None
    
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min)
    
    if not np.allclose(coords[0], coords[-1], atol=1e-3):
        coords = np.vstack([coords, coords[0:1]])
    
    leading_edge_idx = np.argmin(coords[:, 0])
    trailing_edge_idx = np.argmax(coords[:, 0])
    
    y_center = (coords[:, 1].max() + coords[:, 1].min()) / 2
    
    upper_path = []
    lower_path = []
    
    n = len(coords)
    start_idx = leading_edge_idx
    
    for i in range(n):
        idx = (start_idx + i) % n
        point = coords[idx]
        
        if i < n // 2:
            if point[1] >= y_center:
                upper_path.append(point)
            else:
                lower_path.append(point)
        else:
            if point[1] >= y_center:
                upper_path.append(point)
            else:
                lower_path.append(point)
    
    if len(upper_path) == 0 or len(lower_path) == 0:
        return None
    
    upper_path = np.array(upper_path)
    lower_path = np.array(lower_path)
    
    upper_sorted = upper_path[np.argsort(upper_path[:, 0])]
    lower_sorted = lower_path[np.argsort(-lower_path[:, 0])]
    
    if len(upper_sorted) == 0 or len(lower_sorted) == 0:
        return None
    
    x_normalized = np.concatenate([1.0 - upper_sorted[:, 0], lower_sorted[:, 0]])
    y_normalized = np.concatenate([upper_sorted[:, 1], lower_sorted[:, 1]])
    
    y_center_normalized = (y_normalized.max() + y_normalized.min()) / 2
    y_range_normalized = y_normalized.max() - y_normalized.min()
    
    if y_range_normalized > 1e-6:
        y_normalized = (y_normalized - y_center_normalized) / y_range_normalized * 0.15
    
    normalized = np.column_stack([x_normalized, y_normalized])
    
    sorted_indices = np.argsort(normalized[:, 0])
    normalized = normalized[sorted_indices]
    
    if not np.allclose(normalized[0, 0], 1.0, atol=0.01):
        normalized = np.vstack([np.array([[1.0, normalized[0, 1]]]), normalized])
    if not np.allclose(normalized[-1, 0], 1.0, atol=0.01):
        normalized = np.vstack([normalized, np.array([[1.0, normalized[-1, 1]]])])
    
    return normalized

def save_airfoil_dat(coords, output_path, name="Generated Airfoil"):
    if coords is None or len(coords) == 0:
        return False
    
    try:
        with open(output_path, 'w') as f:
            f.write(f"{name}\n")
            for coord in coords:
                f.write(f"       {coord[0]:.5f}     {coord[1]:.5f}\n")
        return True
    except:
        return False

def generate_and_select_best_airfoils(generator, discriminator, device, num_candidates=100, num_best=9, latent_dim=100, output_dir="DCGAN_result"):
    generator.eval()
    discriminator.eval()
    
    print(f"Generating {num_candidates} candidate airfoils...")
    
    candidates = []
    scores = []
    
    with torch.no_grad():
        for batch_start in range(0, num_candidates, 32):
            batch_size = min(32, num_candidates - batch_start)
            noise = torch.randn(batch_size, latent_dim).to(device)
            generated_images = generator(noise)
            generated_binary = (generated_images.cpu().numpy() > 0).astype(np.float32)
            
            fake_output = discriminator(generated_images)
            discriminator_scores = torch.sigmoid(fake_output).cpu().numpy().flatten()
            
            for i in range(batch_size):
                img = generated_binary[i, 0, :, :]
                
                _, labels = cv2.connectedComponents(img.astype(np.uint8))
                num_components = np.max(labels)
                continuity_score = 1.0 / num_components if num_components > 0 else 0
                
                combined_score = 0.7 * continuity_score + 0.3 * discriminator_scores[i]
                
                candidates.append({
                    'image': img,
                    'noise': noise[i].cpu(),
                    'continuity': continuity_score,
                    'discriminator': discriminator_scores[i],
                    'combined': combined_score
                })
                scores.append(combined_score)
    
    sorted_indices = np.argsort(scores)[::-1]
    best_indices = sorted_indices[:num_best]
    
    print(f"\nSelected {num_best} best airfoils:")
    for idx, best_idx in enumerate(best_indices):
        candidate = candidates[best_idx]
        print(f"  Airfoil {idx+1}: Continuity={candidate['continuity']:.4f}, Discriminator={candidate['discriminator']:.4f}, Combined={candidate['combined']:.4f}")
    
    best_images = []
    best_contours = []
    
    for idx, best_idx in enumerate(best_indices):
        candidate = candidates[best_idx]
        img = candidate['image']
        
        contour_points = extract_contour_from_image(img)
        if contour_points is not None:
            resampled = resample_contour_to_points(contour_points, num_points=200)
            if resampled is not None:
                normalized = normalize_airfoil_coordinates(resampled)
                if normalized is not None:
                    best_images.append(img)
                    best_contours.append(normalized)
    
    if len(best_images) == 0:
        print("Warning: No valid contours extracted from generated images")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(min(9, len(best_images))):
        ax = axes[i // 3, i % 3]
        ax.imshow((best_images[i] * 255).astype(np.uint8), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Best Airfoil {i + 1}')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'best_9_airfoils.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    dat_dir = os.path.join(output_dir, 'airfoil_coordinates')
    os.makedirs(dat_dir, exist_ok=True)
    
    print(f"\nSaving {len(best_contours)} airfoil coordinate files...")
    for i, coords in enumerate(best_contours):
        dat_path = os.path.join(dat_dir, f'generated_airfoil_{i+1:02d}.dat')
        name = f"Generated_Airfoil_{i+1:02d}"
        if save_airfoil_dat(coords, dat_path, name):
            print(f"  Saved: {dat_path}")
        else:
            print(f"  Failed to save: {dat_path}")
    
    print(f"\nBest airfoils saved to: {dat_dir}")


if __name__ == "__main__":
    OUTPUT_DIR = "DCGAN_result"
    if not os.path.isabs(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    
    generator, discriminator = train_gan()
    
    if generator is not None and discriminator is not None:
        model_path = os.path.join(OUTPUT_DIR, 'airfoil_generator_final.pth')
        generate_new_airfoils(model_path, OUTPUT_DIR, device)
        evaluate_generated_airfoils(generator, device)
        generate_and_select_best_airfoils(generator, discriminator, device, num_candidates=100, num_best=9, latent_dim=100, output_dir=OUTPUT_DIR)
        print("Training and evaluation completed.")