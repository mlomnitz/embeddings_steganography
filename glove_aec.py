# Standard python packages
import re
from random import sample
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
    def save_example(self, message='', n_images=0, save_path='./Example'):
        reco_message = ''
        converted = self.convert_message(message=message,
                                         n_images=n_images)
        row = 0
        for idx, cover in enumerate(self.coverloader):
            if idx == len(converted)//self.n_words:
                break
            lower = idx*self.n_words
            upper = (idx+1)*self.n_words
            original, covered = self.hide_message(cover, converted[lower:upper])
            reco_message += '{}'.format(self.uncover_message(covered))
            row += len(self.uncover_message(covered))
            if row > 60:
                reco_message += '\n'
                row = 0
    
            fig = make_fig(original[0], covered[0], reco_message)
            fig.savefig('{}/{}.png'.format(save_path, idx))
            
    #
    def convert_message(self, message='', n_images=0):
        w_list = re.findall(r"[\w']+|[.,!?;]", message)
        pad_length = 5*max( (len(w_list)-1)//5+1, n_images)-len(w_list)
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
            return cover, self.Hnet(cat_image)
    #
    
    def uncover_message(self, image=None):
        with torch.set_grad_enabled(False):
            uncovered = self.Rnet(image)
            encripted_message = self.decoder(uncovered)
            message = ''
            for word in encripted_message:
                embedding = word.detach().cpu().numpy().reshape((50,))
                predicted = torch.from_numpy(
                    cos_matrix_multiplication(self.vectors, embedding))
                message += '{} '.format(self.words[predicted.argmax()])
            return message
