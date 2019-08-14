# File handling
import bcolz
import pickle
from random import sample
from collections import Counter
import re
import functools
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class StegEncoderDecoder():
    def __init__(self, embedding_path='', n_words=20, encoder_path=None,
                 decoder_path=None, device):
        """ Encoder and Decoder object
        Parameters
        ----------
        embeddings_path : str
            Path to load the embeddings dictionary for vocabulary
        n_words : int
            Number of words to embed in each message.  Must match loaded model
            architecture
        encoder_path : str
            Path to encoder model weights
        decoder_path : str
            Path to decoder model weights
        device : torch.device
            Device to run on

        Returns
        -------
        """
        embeddings = bcolz.open('{}/6B.50.dat'.format(embedding_path))[:]
        self.vectors = torch.from_numpy(embeddings).float()
        self.words = pickle.load(
            open('{}/6B.50_words.pkl'.format(embedding_path), 'rb'))
        self.word2idx = pickle.load(
            open('{}/6B.50_idx.pkl'.format(embedding_path), 'rb'))
        self.vocab_size = len(self.vectors)
        self.n_words = n_words
        self.encoder = Encoder(n_words, upsample=True).to(device)
        self.decoder = Decoder(n_words).to(device)
        if encoder_path is not None:
            chpt = torch.load(encoder_path)
            self.encoder.load_state_dict(chpt['state_dict'])
        if decoder_path is not None:
            chpt = torch.load(decoder_path)
            self.decoder.load_state_dict(chpt['state_dict'])

    def message_2_labels(self, message='', replace_unknown=False):
        """ Converts message into a list of tokens of their associated
        embeddings indices
        Parameters
        ----------
        messagge : str
            Message to convert
        replace_unkown : bool
            Swith to either replace words outside vocabulary or quit

        Returns
        -------    
        converted_tokens : list(int)
            List of tokens representing message
        """  
        message_tokens = re.findall(r"[\w']+|[.,!?;]", message)

        pad_length = self.n_words * \
            ((len(message_tokens)-1)//self.n_words+1)-len(message_tokens)
        converted_tokens = []

        for token in message_tokens:
            try:
                converted_tokens += [self.word2idx[token]]
            except KeyError:
                if replace_unknown is True:
                    converted_tokens += [self.word2idx['blank']]
                else:
                    print(
                        'Token for {} does not exist. Please replace token in\
                        message and retry.'.format(token))
        if pad_length > 0:
            pad_sequence = [self.word2idx['blank']]*pad_length
            self.words[self.word2idx['blank']] = ''
            insert_locations = sample(
                range(len(converted_tokens) + len(pad_sequence)),
                len(pad_sequence))
            inserts = dict(zip(insert_locations, pad_sequence))
            message_iterator = iter(converted_tokens)
            converted_tokens[:] = [inserts[pos] if pos in inserts else next(
                message_iterator) for pos in range(len(converted_tokens) + len(pad_sequence))]

        return converted_tokens

    def label_2_embeddings(self, labels, embedding_size=50):
        """ Converts a list of tokens int embeddings
        Parameters
        ----------
        labels : list(int)
            List of tokens associating words to embeddings
        embedding_size : int
            Length of each embedding

        Returns
        -------    
        embedding : torch.tensor
            Torch tensor containg the batch of embeddings
        """  
        embedding = torch.FloatTensor(labels.shape[0], embedding_size).zero_()
        for idx, label in enumerate(labels):
            embedding[idx] = self.vectors[label]

        return embedding


class Hnet():
    def __init__(self, Hnet_path=None, device):
        """ Torch model used to hide one image in another
        Parameters
        ----------
        Hnet_path : str
            Path to the hidding net weights
        device : torch.device
            Device to run processes

        Returns
        -------    
        """  
        self.model = UnetGenerator(input_nc=6, output_nc=3, num_downs=7,
                                   output_function=nn.Sigmoid).to(device)
        if Hnet_path is not None:
            self.model.load_state_dict(torch.load(Hnet_path))
            self.model.eval()
        self.device = device

    def hide_message(self, encoder, embedding, cover_img):
        """ Hide message embeddings in a cover image
        Parameters
        ----------
        encoder : torch.nn.Module
            Torch encoder module
        embedding : torch.tensor
            Torch embeddings tensor
        cover_img : torch.tensor
            Cover image tensor

        Returns
        -------    
        cover_img : torch.tensor
            Original torch tensor for the cover image
        message_image : torch.tensor
            Encoded message image
        container_img : torch.tensor
            Container image with hidden message
        """  
        with torch.set_grad_enabled(False):
            message_image = encoder(embedding.to(self.device))
            cat_image = torch.cat(
                [cover_img.to(self.device), message_image], dim=1)
            return cover_img, message_image, self.model(cat_image)


class Rnet():
    def __init__(self, Rnet_path=None, device):
        """ Recovery network used to reveal hidden images
        Parameters
        ----------
        Rnet_path : str
            Path to the recovery network model weights
        device : torch.device
            Device to run processes
        Returns
        -------    
        """  
        self.model = RevealNet(output_function=nn.Sigmoid).to(device)
        if Rnet_path is not None:
            self.model.load_state_dict(torch.load(Rnet_path))
            self.model.eval()
        self.device = device

    def recover_message(self, Rnet, encoder_decoder, image):
        """ Recovers and decodes a message from a container image
        Parameters
        ----------
        Rnet : torch.nn.Module
            Torch model for image recovery
        encoder_decoder : torch.nn.Module
            Torch model to decode message image back into embeddings
        image : torch.tensor
            Container image

        Returns
        -------    
        message : str
            Revealed and decoded message
        """  
        with torch.set_grad_enabled(False):
            encoder_decoder.decoder.eval()
            message = []
            image_tensor = image.to(self.device)
            encoding = Rnet(image_tensor)
            recovered_message = encoder_decoder.decoder(encoding)
            recovered_message = recovered_message.reshape((20, 50))
            for i_word in range(20):
                embedding = recovered_message[i_word].reshape((1, 50))
                predicted = nn.CosineSimilarity()(
                    encoder_decoder.vectors.to(self.device), embedding)
                predicted = encoder_decoder.words[predicted.argmax()]
                message.append(predicted)
            return " ".join(message)


# BASIC CONSTRUCTION CLASSES FOR ABOVE

class Encoder(nn.Module):
    def __init__(self, n_words=1, embeddings_dim=50, upsample=False):
        """ Network used to encode embeddings into images
        Parameters
        ----------
        n_words : int
            Number of words per image
        embeddings_dim : int
            Dimensions of each embedding
        upsample : bool
            Boolean to switch between upsampling layer and transpose
            convolution
        Returns
        -------    
        """  
        super(Encoder, self).__init__()
        self.n_words = n_words
        self.embeddings_dim = embeddings_dim
        # Model sizes
        in_sizes = [256, 256, 128, 64]
        inter_sizes = [256, 256, 128, 64]
        out_sizes = [256, 128, 64, 3]
        #
        self.down_path = nn.ModuleList()
        if upsample:
            for idx in range(len(in_sizes)):
                self.down_path.append(UpsampleBlock(in_sizes[idx],
                                                    out_sizes[idx]))
        else:
            for idx in range(len(in_sizes)):
                self.down_path.append(DeconvBlock(in_sizes[idx],
                                                  inter_sizes[idx],
                                                  out_sizes[idx]))
        #
        self.fc = nn.Linear(embeddings_dim*n_words, 65536)

    def forward(self, x):
        """ Forward pass for encoder 
        Parameters
        ----------
        x : torch.tensor
            Tensor to process

        Returns
        -------    
        prediction : torch.tensor
            Prediction torch tensor
        """  
        x = x.view(-1, self.embeddings_dim*self.n_words)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 256, 16, 16)
        for layer in self.down_path:
            x = layer(x)
        return F.sigmoid(x)


class Decoder(nn.Module):
    def __init__(self, n_words=1, embeddings_dim=50):
        """ Torch model used for decoding messages from images
        Parameters
        ----------
        n_words : int
            Number of words per message
        emvedding_dim : int
            Dimension of embeddings

        Returns
        -------    
        """  
        super(Decoder, self).__init__()
        self.n_words = n_words
        # Model sizes
        insize = [3, 64, 128, 256]
        outsize = [64, 128, 256, 256]
        #
        self.up_path = nn.ModuleList()
        for idx in range(len(insize)):
            self.up_path.append(ConvBlock(in_channels=insize[idx],
                                          out_channels=outsize[idx]))
        self.fc = nn.Linear(65536, embeddings_dim*self.n_words)

    def forward(self, x):
        """ Forward pass through model
        Parameters
        ----------
        x : torch.tensor
            Input torch tensor images (B, 3, H, W)

        Returns
        -------    
        embeddings : torch.tensor
            Decoded embeddings tensor (B*n_words, embeddings_dim) 
        """  
        for layer in self.up_path:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(x.shape[0]*self.n_words, -1)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size):
        """ Upsample block for encoding
        Parameters
        ----------
        in_size : int
            Input size for convolution filters
        out_size : int
            Output nunber of filters for convolution

        Returns
        -------    
        """  
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Input tensor to process (B, input_size, H, W)

        Returns
        -------    
        x : torch.tensor
            Processed tensor (B, output_size, 2*H, 2*W)
        """  
        return self.upsample_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_size, inter_size, out_size):
        """ Transpose convolution block for encoder
        Parameters
        ----------
        in_size : int
            Input size for convolution filters
        inter_size : int
            Number of filters for intermediate convolution
        out_size : int
            Output nunber of filters for convolution

        Returns
        -------    
        """  
        super(DeconvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, inter_size, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.deconv = nn.ConvTranspose2d(inter_size, out_size, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

    def forward(self, x):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Input tensor to process (B, input_size, H, W)

        Returns
        -------    
        x : torch.tensor
            Processed tensor (B, output_size, 2*H, 2*W)
        """  
        x = self.conv(x)
        x = self.deconv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """ Convolution block for decoder
        Parameters
        ----------
        in_size : int
            Input size for convolution filters
        out_size : int
            Output nunber of filters for convolution

        Returns
        -------   
        """  
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Input tensor to process (B, input_size, H, W)

        Returns
        -------    
        x : torch.tensor
            Processed tensor (B, output_size, H/2, W/2)
        """          
        return self.block(x)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 output_function=nn.Sigmoid):
        """ Unet Model definition for hidding and recovery networks.  Adapted 
        from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        Parameters
        ----------
        input_nc : int
            Input number of channels
        output_nc : int
            Output number of channels
        num_downs : int
            Number of down and up blocks
        ngf : int
            Number of in and out filters in the unet blocks
        norm_layer : torch.nn.Module
           Torch nomralization layer to use in blocks
        use_droput : bool
            Boolean value to turn dropout on or off
        output_function : torch.nn.Module
            Output activation function for last layer
        Returns
        -------    
        """  
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

    def forward(self, input):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Input torch tensors (B, input_nc, H, W)
        Returns
        -------    
        x : torch.tensor
            Processed tensor size (B, output_nc, H, W)
        """  
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        """ Basic Unet skip blocks
        Parameters
        ----------
        outer_nc : int
            Outer most number of channels
        inner_nc : int
            Inner number of channels
        input_nc : int
            Input number of channels
        submodule : torch.nn.Module
            Nested submodule in model
        outermost : bool
            Outermost block?  Used to specify if skip connections and such
        innermost : bool
            Innermost block? Used to specify if skip connections and such
        norm_layer : torch.nn.Module
            Normalization layer to use in block
        use_dropout : bool
            Turn dropout on or off
        output_function : torch.nn.Module
            Activation function to apply on output
        Returns
        -------    
        """  
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Tensor to process

        Returns
        -------    
        x : torch.tensor
            Processed tensor
        """   
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        """ Reaveal network definitio
        Parameters
        ----------
        nc : int
            Number of channles for output and input
        nhf : int
            Used to define the number of filters in intermediate steps
        output_function : torch.nn.Module
            Activation function to apply on output

        Returns
        -------    
        """   
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward(self, input):
        """ Forward pass
        Parameters
        ----------
        input : torch.tensor
            Input tensor to process size (B, 3, H, W)

        Returns
        -------    
        output : torch.tensor (B, 3, H, W)
        """   
        output = self.main(input)
        return output
