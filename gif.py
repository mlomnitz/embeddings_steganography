
import subprocess
import models
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from wand.image import Image as wandImage
from apng import APNG


def process_gif(path=''):
    """ Processes gif and saves individual frames to png

    Parameters
    ----------
    path : str
        Path to input gif to process

    Returns
    -------
    """
    with wandImage(filename=path) as img:
        img.coalesce()
        img.format = 'png'
        img.save(filename='./gif/frame%05d.png')
        

def save_frame(frame, output):
    """ Saves png frames from an input torch tensor

    Parameters
    ----------
    frame : torch.tensor
        Torch tensor for the image to be saved to png dimensions (3xHxW)

    Returns
    -------
    output : str
        Path to location to save the frame as png
    """
    frame = frame.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(np.uint8(255*(frame)), 'RGB')
    img.save(output)
    

def frames_to_gif(save_path='./message.png'):
    """ Merges frames into an animated png (APNG)

    Parameters
    ----------
    save_path : str
        Path describing the location to save the APNG

    Returns
    -------    

    """
    n_frames = len(os.listdir('./gif'))
    im = APNG()
    for idx in range(n_frames):
        im.append_file('./gif/frame{0:05d}.png'.format(idx), delay=50)
    im.save(save_path)
    

class GIFDataset(Dataset):
    def __init__(self, transform=None, folder_path=''):
        """ GIF dataset used to load frames into torch tensors
        Parameters
        ----------
        transform : torch.torchvision.transforms
            Transformation for the GIF dataset frames
        folder_path : str
            Path to folder where the frames to be loaded are located

        Returns
        -------
        """    
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
        
        self.dataset = sorted(os.listdir(folder_path))
        self.folder_path = folder_path
                                  
    def __getitem__(self, index):
        """ Returns a data-point for processing
        Parameters
        ----------
        index : int
            Index for data-point to load

        Returns
        -------    
        image : torch.tensor
            torch tensor with data-point for processing
        """    
        image = Image.open(self.folder_path + '/' + self.dataset[index]).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        """ Return the length of the data-set
        Parameters
        ----------

        Returns
        -------    
        len : int
           Total number of items in the data-set
        """            
        return len(self.dataset)  

   
def encode_gif(gif_dataset, hider, encoder_decoder, embeddings, device):
    """ Encodes secret message into a gif
    Parameters
    ----------
    gif_dataset : torch.utils.data.Dataset
        Gif dataset class to load images
    hider : torch.nn.Module
        Torch model used to hide images
    encoder_decoder : torch.nn.Module
        Torch models for encoding and decoding messages
    embeddigns : torch.tensor
        Tensor containing embeddings representing the message to hide
    device : torch.device
        Device to run the processes on

    Returns
    -------    
    """    
    gif_loader = DataLoader(gif_dataset, batch_size=1,
                            shuffle=False, num_workers=1)
    for idx, cover in enumerate(gif_loader):
        lower = idx*20
        upper = (idx+1)*20
        if upper > len(embeddings):
            end_token = [encoder_decoder.word2idx['______________________________________']]
            blank_tokens = [encoder_decoder.word2idx['blank']]*19 + end_token
            blank_embeddings = encoder_decoder.label_2_embeddings(torch.LongTensor(blank_tokens))
            cover, secret, container = hider.hide_message(encoder_decoder.encoder, blank_embeddings, cover, device=device)
            save_frame(cover[0], output='./gif/frame{0:05d}.png'.format(idx))
        else:
            cover, secret, container = hider.hide_message(encoder_decoder.encoder, embeddings[lower:upper], cover, device=device)
            fig = save_frame(container[0], output='./gif/frame{0:05d}.png'.format(idx))
    frames_to_gif()
    
   
def decode_gif(reveal_net, encoder_decoder, gif_path='', device):
    """ Decodes a gif containing secret messages
    Parameters
    ----------
    reveal_net : torch.nn.Module
        Torch model that reveals hidden images
    encoder_decoder : torch.nn.Module
        Torch models for encoding and decoding messages
    gif_path : str
        Path to gif to decode
    device : torch.device
        Device to run processes

    Returns
    -------    
    message : str
        Recovered message
    """
    process_gif(gif_path)
    encoded_frames = GIFDataset(folder_path='./gif')
    frameloader = DataLoader(encoded_frames, batch_size=1, shuffle=False,
                             num_workers=1)
    message = ''
    tot_frames = len(os.listdir('./gif'))
    for idx, frame in enumerate(frameloader):
        if idx == tot_frames:
            break
        else:
            frame = frame.to(device)
            revealed = reveal_net.recover_message(reveal_net.model,
                                                  encoder_decoder, frame,
                                                  device=device)
            message += ' '+revealed
    return message
