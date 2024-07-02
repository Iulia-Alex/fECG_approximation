import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
import os
import torch
import numpy as np
from prepare_data import create_spectrogram

class SpectrogramDataset(Dataset):
    def __init__(self, root_path):
        self.files = [os.path.join(root_path, file) for file in os.listdir(root_path) if file.endswith('.mat')]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # torchvision.ops.Permute([1, 2, 0]), 
            transforms.Resize((128, 128))
        ])
        
    def __getitem__(self, idx):
        current_mat = self.files[idx]
        # Spectrogram computing
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "ecg")
        spect_dir = os.path.join(current_dir, "data/processed/spectrogram")
        input_spec, output_spec, _, _ = create_spectrogram(data_path, idx)
        input_spec = np.moveaxis(input_spec, 0, -1)
        input_spec = self.transform(input_spec)
        output_spec = np.moveaxis(output_spec, 0, -1)
        output_spec = self.transform(output_spec)
        return input_spec, output_spec

    def __len__(self):
        return len(self.files)
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "ecg")

    dset = SpectrogramDataset(data_path)
    
    for i in range(1,2):
        x, y = dset[i]
        print(x.shape, y.shape)