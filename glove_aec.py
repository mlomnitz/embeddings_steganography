# Standard python packages
import os
import re
import shutil
from random import sample
import numpy as np
# PyTorch imports
import torch
import torch.nn as nn
import torchvision
# Project imports
from RevealNet import RevealNet
from HidingUNet import UnetGenerator
from utils import *
from embeddings import glove
from gif_dataset import GIF_dataset
# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class encoder_decoder(): 
    def __init__(self, encoder=None, decoder=None, glove_path='../GLoVE',
                 dis_path='../PyTorch-Deep-Image-Steganography/checkPoint',
                 cover_dir='./GIFsource/frames',
                 cover_transform=None):
        embeddings = glove(glove_path=glove_path)
        self.vectors = embeddings.vectors
        self.words = embeddings.words
        self.word2idx = embeddings.word2idx
        self.vocab_size = len(self.vectors)

        #
        self.Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7,
                                  output_function=nn.Sigmoid).to(device)
        self.Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
        Hchpt = 'epoch_73,sumloss=0.000447,Hloss=0.000258'
        Rchpt = 'epoch_73,sumloss=0.000447,Rloss=0.000252'
        self.Hnet.load_state_dict(torch.load('{}/netH_{}.pth'.format(dis_path,
                                                                     Hchpt)))
        self.Hnet.eval()
        #
        self.Rnet.load_state_dict(torch.load('{}/netR_{}.pth'.format(dis_path,
                                                                     Rchpt)))
        self.Rnet.eval()
        #
        self.encoder = encoder.to(device)
        self.encoder.eval()
        #
        self.decoder = decoder.to(device)
        self.decoder.eval()
        #
        self.n_words = decoder.n_words
        #
        self.coverloader = self.init_cover(cover_dir, cover_transform)
        
    #
    @staticmethod
    def init_cover(cover_dir='', cover_transform=None):
        
        coverset = GIF_dataset(transform=cover_transform, folder_path=cover_dir)
        coverloader = torch.utils.data.DataLoader(coverset, batch_size=1,
                                                  shuffle=False, num_workers=1)
        return coverloader
    
    #
    def label_2_embeddings(self, labels, embedding_size=50):
        embedding = torch.FloatTensor(labels.shape[0], embedding_size).zero_()
        for idx, label in enumerate(labels):
            embedding[idx] = torch.from_numpy(self.vectors[label])
            
        return embedding

    #
    def random_insert_seq(self, lst, seq):
        insert_locations = sample(range(len(lst) + len(seq)), len(seq))
        inserts = dict(zip(insert_locations, seq))
        input = iter(lst)
        lst[:] = [inserts[pos] if pos in inserts else next(input)
                  for pos in range(len(lst) + len(seq))]

    #
    def convert_message(self, message='', n_images=0):
        w_list = re.findall(r"[\w']+|[.,!?;]", message)
        pad_length = self.n_words*max( (len(w_list)-1)//self.n_words+1, n_images)-len(w_list)
        converted = []
        for word in w_list:
            converted += [self.word2idx[word]]
        if pad_length > 0:
            seq = [self.word2idx['blank']]*pad_length
            self.words[self.word2idx['blank']] = ''
            self.random_insert_seq(converted, seq)     
        return converted
    
    #
    def hide_message(self, cover=None, message=None):
        with torch.set_grad_enabled(False):
            image = self.encoder(self.label_2_embeddings(
                torch.tensor(message)).to(device))

            cat_image = torch.cat([cover.to(device), image], dim=1)
            return cover, image, self.Hnet(cat_image)
    
    #
    def uncover_message(self, image=None):
        with torch.set_grad_enabled(False):
            revealed = self.Rnet(image)
            encripted_message = self.decoder(revealed)
            message = ''
            for word in encripted_message:
                embedding = word.detach().cpu().numpy().reshape((50,))
                predicted = torch.from_numpy(
                    cos_matrix_multiplication(self.vectors, embedding))
                message += '{} '.format(self.words[predicted.argmax()])
            return message, revealed

    #
    def encode_message(self, message='', n_images=0, save_path='./Example'):
        converted = self.convert_message(message=message, n_images=n_images)
        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        #
        for idx, cover in enumerate(self.coverloader):
            lower = idx*self.n_words
            upper = (idx+1)*self.n_words
            if upper > len(converted):
                break
            cover, secret, container = self.hide_message(cover, converted[lower:upper])
            fig = save_frame(container[0], output='{}/frame{}.png'.format(temp_dir, idx))

        frames_to_gif(source_dir=temp_dir,
                      save_path='{}/message.png'.format(save_path))
        shutil.rmtree(temp_dir)

    #
    def decode_message(self, gif='', save_path='./message.txt'):
        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        gif_to_frames(gif=gif, save_path=temp_dir)
        
        frameset = GIF_dataset(folder_path=temp_dir)
        frameloader = torch.utils.data.DataLoader(frameset, batch_size=1,
                                                  shuffle=False, num_workers=1)
        message = ''
        tot_frames = len(os.listdir(temp_dir))
        for idx, frame in enumerate(frameloader):
            if idx == tot_frames:
                break
            frame = frame.to(device)
            this_frame, revealed = self.uncover_message(frame)
            message += ' '+this_frame
        shutil.rmtree(temp_dir)
        text_file = open(save_path, 'w')
        text_file.write(message)
        text_file.close()
        return message
        
    #
    def run_benchmark(self, message='', n_images=0, save_path='./Example',
                      max_words=0):
        #
        reco_message = ''
        converted = self.convert_message(message=message,
                                         n_images=n_images)
        if max_words > 0:
            converted = converted[:max_words]
        row = 0
        for idx, cover in enumerate(self.coverloader):
            if idx == len(converted)//self.n_words:
                break
            lower = idx*self.n_words
            upper = (idx+1)*self.n_words
            cover, secret, container = self.hide_message(cover, converted[lower:upper])
            this_frame, revealed = self.uncover_message(container)
            reco_message += ' '+this_frame
            row += len(this_frame)
            if row > 50:
                reco_message += '\n'
                row = 0            
            #
            frame_error, frame_psnr = image_metrics(cover[0].cpu(),
                                                    container[0].cpu())
            if idx == 0:
                pix_error = frame_error
                psnr = frame_psnr
            else:
                pix_error = np.concatenate([pix_error, frame_error])
                psnr = np.concatenate([psnr, frame_psnr])
            #
            fig = make_fig(cover[0].cpu(), container[0].cpu(), secret[0].cpu(),
                           revealed[0].cpu(), pix_error, psnr, reco_message)
            fig.savefig('{}/{}.png'.format(save_path, idx))
