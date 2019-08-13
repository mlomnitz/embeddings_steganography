##Note: messages that are longer than the number of frames in the GIF (each frame holds 20 words) generally do not decode well.

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


def process_gif(path = ''):
    with wandImage(filename=path) as img:
        img.coalesce()
        img.format = 'png'
        img.save(filename='./gif/frame%05d.png')
        
def save_frame(frame, output):
    frame = frame.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(np.uint8(255*(frame)), 'RGB')
    img.save(output)
    
def frames_to_gif(save_path='./message.png'):
    n_frames = len(os.listdir('./gif'))
    im = APNG()
    for idx in range(n_frames):
        im.append_file('./gif/frame{0:05d}.png'.format(idx), delay=50)
    im.save(save_path)
    
class GIFDataset(Dataset):
    
    def __init__(self, transform=None, folder_path=''):
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
        image = Image.open(self.folder_path + '/' + self.dataset[index]).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataset)  


    
def encode_gif(gif_dataset, hider, encoder_decoder, embeddings, device = 'cuda'):
    gif_loader = DataLoader(gif_dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)
    for idx, cover in enumerate(gif_loader):
        lower = idx*20
        upper = (idx+1)*20
        if upper > len(embeddings):
            end_token = [encoder_decoder.word2idx['______________________________________']]
            blank_tokens = [encoder_decoder.word2idx['blank']]*19 + end_token
            blank_embeddings = encoder_decoder.label_2_embeddings(torch.LongTensor(blank_tokens))
            cover, secret, container = hider.hide_message(encoder_decoder.encoder, blank_embeddings, cover, device = device)
            save_frame(cover[0], output='./gif/frame{0:05d}.png'.format(idx))
        else:
            cover, secret, container = hider.hide_message(encoder_decoder.encoder, embeddings[lower:upper], cover, device = device)
            fig = save_frame(container[0], output='./gif/frame{0:05d}.png'.format(idx))
    frames_to_gif()
    

    
def decode_gif(reveal_net, encoder_decoder, gif_path = '', device = 'cuda'):
    process_gif(gif_path)
    encoded_frames = GIFDataset(folder_path = './gif')
    frameloader = DataLoader(encoded_frames, batch_size=1,
                                                  shuffle=False, num_workers=1)
    message = ''
    tot_frames = len(os.listdir('./gif'))
    for idx, frame in enumerate(frameloader):
        if idx == tot_frames:
                break
        else:
            frame = frame.to(device)
            revealed = reveal_net.recover_message(reveal_net.model, encoder_decoder, frame, device=device)
            message += ' '+revealed
    return message
