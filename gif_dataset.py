# Python imports
import os
from PIL import Image
# PyTorch imports
from torch.utils.data import Dataset
from torchvision import transforms


class GIF_dataset(Dataset):
    def __init__(self, transform=None, folder_path=''):
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        # Create dicts

        def sort(path=''):
            dataset = []
            for idx in range(len(os.listdir(path))):
                dataset.append('{}/frame{}.png'.format(path, idx+1))
            return dataset
        self.datasetid_to_filepath = sort(folder_path)
                    
    def __getitem__(self, index):
        image = Image.open(self.datasetid_to_filepath[index])
        # Add transforms
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.datasetid_to_filepath)
