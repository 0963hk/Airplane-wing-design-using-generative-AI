import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import torch


def read_airfoil_data(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        airfoil_name = lines[0].strip()
        coords = []
        for line in lines[1:]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue
        coords = np.array(coords)
        if len(coords) < 10 or np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return None, None
        coords = clean_and_ensure_closed(coords)
        if coords is None:
            return None, None
        return airfoil_name, coords
    except Exception:
        return None, None


def clean_and_ensure_closed(coords):
    if len(coords) < 10:
        return None
    coords = coords[~np.isnan(coords).any(axis=1)]
    coords = coords[~np.isinf(coords).any(axis=1)]
    if len(coords) < 10:
        return None
    unique_coords = [coords[0]]
    tolerance = 1e-6
    for i in range(1, len(coords)):
        last_point = unique_coords[-1]
        current_point = coords[i]
        distance = np.sqrt((current_point[0] - last_point[0])**2 + (current_point[1] - last_point[1])**2)
        if distance > tolerance:
            unique_coords.append(current_point)
    if len(unique_coords) < 10:
        return None
    coords = np.array(unique_coords)
    first_point = coords[0]
    last_point = coords[-1]
    distance = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
    if distance > 0.01:
        coords = np.vstack([coords, coords[0:1]])
    return coords


def create_proportional_airfoil_image(coords, image_width=256, image_height=128, line_width=2):
    if coords is None:
        return None
    img = np.zeros((image_height, image_width), dtype=np.uint8)
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    airfoil_width = x_max - x_min
    airfoil_height = y_max - y_min
    scale_x = (image_width - 20) / airfoil_width
    scale_y = (image_height - 20) / airfoil_height
    scale = min(scale_x, scale_y)
    offset_x = (image_width - airfoil_width * scale) / 2
    offset_y = (image_height - airfoil_height * scale) / 2
    pixel_coords = np.zeros_like(coords)
    pixel_coords[:, 0] = (coords[:, 0] - x_min) * scale + offset_x
    pixel_coords[:, 1] = (coords[:, 1] - y_min) * scale + offset_y
    pixel_coords = pixel_coords.astype(int)
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_width - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_height - 1)
    for i in range(len(pixel_coords) - 1):
        pt1 = (pixel_coords[i, 0], image_height - 1 - pixel_coords[i, 1])
        pt2 = (pixel_coords[i + 1, 0], image_height - 1 - pixel_coords[i + 1, 1])
        cv2.line(img, pt1, pt2, 255, line_width)
    return img


def create_proportional_filled_airfoil_image(coords, image_width=256, image_height=128):
    if coords is None:
        return None
    img = np.zeros((image_height, image_width), dtype=np.uint8)
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    airfoil_width = x_max - x_min
    airfoil_height = y_max - y_min
    scale_x = (image_width - 20) / airfoil_width
    scale_y = (image_height - 20) / airfoil_height
    scale = min(scale_x, scale_y)
    offset_x = (image_width - airfoil_width * scale) / 2
    offset_y = (image_height - airfoil_height * scale) / 2
    pixel_coords = np.zeros_like(coords)
    pixel_coords[:, 0] = (coords[:, 0] - x_min) * scale + offset_x
    pixel_coords[:, 1] = (coords[:, 1] - y_min) * scale + offset_y
    pixel_coords = pixel_coords.astype(int)
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_width - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_height - 1)
    adjusted_coords = []
    for x, y in pixel_coords:
        adjusted_coords.append([x, image_height - 1 - y])
    if len(adjusted_coords) > 2:
        contour = np.array(adjusted_coords, dtype=np.int32)
        cv2.fillPoly(img, [contour], 255)
    return img


def create_proportional_sdf_image(coords, image_width=256, image_height=128):
    if coords is None:
        return None
    filled_img = create_proportional_filled_airfoil_image(coords, image_width, image_height)
    if filled_img is None:
        return None
    dist_inside = cv2.distanceTransform(filled_img, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(255 - filled_img, cv2.DIST_L2, 5)
    sdf = dist_outside - dist_inside
    sdf_normalized = cv2.normalize(sdf, None, 0, 255, cv2.NORM_MINMAX)
    return sdf_normalized.astype(np.uint8)


def generate_proportional_airfoil_images():
    output_dirs = {
        'contour_proportional': 'airfoil_images/contour_proportional',
        'filled_proportional': 'airfoil_images/filled_proportional',
        'sdf_proportional': 'airfoil_images/sdf_proportional'
    }
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
    airfoil_dir = 'processed_airfoils'
    image_width = 256
    image_height = 128
    success_count = 0
    for i, filename in enumerate(sorted(os.listdir(airfoil_dir))):
        if filename.endswith('.dat'):
            filepath = os.path.join(airfoil_dir, filename)
            try:
                airfoil_name, coords = read_airfoil_data(filepath)
                if coords is None or len(coords) < 10:
                    continue
                contour_img = create_proportional_airfoil_image(coords, image_width, image_height)
                filled_img = create_proportional_filled_airfoil_image(coords, image_width, image_height)
                sdf_img = create_proportional_sdf_image(coords, image_width, image_height)
                base_name = f"airfoil_{i + 1:04d}"
                if contour_img is not None:
                    cv2.imwrite(os.path.join(output_dirs['contour_proportional'], f"{base_name}.png"), contour_img)
                if filled_img is not None:
                    cv2.imwrite(os.path.join(output_dirs['filled_proportional'], f"{base_name}.png"), filled_img)
                if sdf_img is not None:
                    cv2.imwrite(os.path.join(output_dirs['sdf_proportional'], f"{base_name}.png"), sdf_img)
                success_count += 1
            except Exception:
                continue


def create_comparison_image():
    airfoil_dir = 'airfoil_data'
    example_files = ['airfoil_0001.dat', 'airfoil_0100.dat', 'airfoil_0500.dat']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, filename in enumerate(example_files):
        if i >= 3:
            break
        filepath = os.path.join(airfoil_dir, filename)
        airfoil_name, coords = read_airfoil_data(filepath)
        if coords is None:
            continue
        square_img = create_proportional_airfoil_image(coords, 256, 256)
        proportional_img = create_proportional_airfoil_image(coords, 256, 128)
        axes[0, i].imshow(square_img, cmap='gray')
        axes[0, i].set_title(f'{airfoil_name}\nSquare (256x256)')
        axes[0, i].axis('off')
        axes[1, i].imshow(proportional_img, cmap='gray')
        axes[1, i].set_title(f'{airfoil_name}\nProportional (256x128)')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('airfoil_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


class ProportionalAirfoilDataset:
    def __init__(self, image_dir, metadata_file, image_type='contour_proportional',
                 target_size=(128, 256), transform=None):
        self.image_dir = os.path.join(image_dir, image_type)
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.target_size = target_size
        self.image_type = image_type

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = f"{row['filename']}.png"
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('L')
            if self.target_size and image.size != (self.target_size[1], self.target_size[0]):
                image = image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            img_array = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(img_array).unsqueeze(0)
            if self.transform:
                image_tensor = self.transform(image_tensor)
            features = torch.tensor([
                row['max_thickness'],
                row['max_camber']
            ], dtype=torch.float32)
            return {
                'image': image_tensor,
                'features': features,
                'name': row['name'],
                'id': row['id']
            }
        except Exception:
            blank_image = torch.zeros((1, *self.target_size), dtype=torch.float32)
            blank_features = torch.zeros(2, dtype=torch.float32)
            return {
                'image': blank_image,
                'features': blank_features,
                'name': 'error',
                'id': -1
            }


def prepare_proportional_gan_dataset(image_type='contour_proportional', target_size=(128, 256)):
    image_dir = f'airfoil_images1/{image_type}'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    images = []
    for filename in image_files:
        try:
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert('L')
            if target_size:
                img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
        except Exception:
            continue
    images = np.array(images)
    images = images.reshape(images.shape[0], 1, target_size[0], target_size[1])
    np.save(f'airfoil_{image_type}_gan.npy', images)
    return images


if __name__ == "__main__":
    generate_proportional_airfoil_images()
    create_comparison_image()
    prepare_proportional_gan_dataset('contour_proportional', (128, 256))
    prepare_proportional_gan_dataset('filled_proportional', (128, 256))
