import os
import torch
from torchvision import datasets, transforms

class Dataset:
    def __init__(self, data_dir, _batch_size, train_split=0.8):
        super(Dataset, self).__init__()
        
        data_transform = transforms.Compose([
            transforms.Resize((150, 150)),  # Resize the images to 150x150
            transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        full_dataset = datasets.ImageFolder(data_dir, transform=data_transform)
        
        # Calculate the number of samples for the training and testing split
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        # Split the dataset into train and test subsets
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)
