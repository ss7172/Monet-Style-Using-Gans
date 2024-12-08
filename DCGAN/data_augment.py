import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Define augmentation transforms with proper normalization
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Updated for 3 channels
])

class MonetDataset(Dataset):
    def __init__(self, image_dir, transform=None, augment_times=10, save_augmented=False, save_dir='augmented_images'):
        self.image_dir = image_dir
        self.transform = transform
        self.augment_times = augment_times
        self.save_augmented = save_augmented
        self.save_dir = save_dir
        self.image_files = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if self.save_augmented:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.image_files) * self.augment_times
    
    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        img_path = self.image_files[img_idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        
        if self.save_augmented:
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                augmented_filename = f"{base_name}_aug_{idx}.png"
                save_path = os.path.join(self.save_dir, augmented_filename)
                
                unnormalize = transforms.Normalize(
                    mean=[-1, -1, -1],
                    std=[2, 2, 2]
                )
                image_unnormalized = unnormalize(image_transformed)
                image_to_save = transforms.ToPILImage()(image_unnormalized)
                image_to_save.save(save_path)
            except Exception as e:
                print(f"Error saving image {augmented_filename}: {e}")
        
        return image_transformed

if __name__ == '__main__':
    dataroot = "data"  # Path to your Monet images directory
    
    # Create the dataset with augmentation and saving enabled
    dataset = MonetDataset(
        image_dir=dataroot, 
        transform=augmentation_transforms, 
        augment_times=70, 
        save_augmented=True,  # Enable saving
        save_dir='augmented_images'  # Specify save directory
    )
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=12  # Adjust based on your CPU cores
    )
    
    # Iterate through the dataset to trigger augmentation and saving
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(f"Processed {i * 32} / {len(dataset)} images")
        # Optionally, add a limit to avoid processing all 98,000 images at once
        # if i * 32 >= 10000:
        #     break
    
    print("Augmentation and saving completed.")
