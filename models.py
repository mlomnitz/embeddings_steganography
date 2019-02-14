import torch.nn as nn
import torch.nn.functional as F


class upsample_block(nn.Module):
    def __init__(self, in_size, out_size):
        super(upsample_block, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        return self.upsample_block(x)


class deconv_block(nn.Module):
    def __init__(self, in_size, inter_size, out_size):
        super(deconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, inter_size, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.deconv = nn.ConvTranspose2d(inter_size, out_size, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
    

class encoder(nn.Module):
    def __init__(self, n_words=1, embeddings_dim=50, upsample=False):
        super(encoder, self).__init__()
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
                self.down_path.append(upsample_block(in_sizes[idx],
                                                     out_sizes[idx]))
        else:
            for idx in range(len(in_sizes)):
                self.down_path.append(deconv_block(in_sizes[idx],
                                                   inter_sizes[idx],
                                                   out_sizes[idx]))
        #
        self.fc = nn.Linear(embeddings_dim*n_words, 65536)
    
    def forward(self, x):
        x = x.view(-1, self.embeddings_dim*self.n_words)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 256, 16, 16)
        for layer in self.down_path:
            x = layer(x)
        return F.sigmoid(x)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

    
class decoder(nn.Module):
    def __init__(self, n_words=1, embeddings_dim=50):
        super(decoder, self).__init__()
        self.n_words = n_words
        # Model sizes
        insize = [3, 64, 128, 256]
        outsize = [64, 128, 256, 256]
        #
        self.up_path = nn.ModuleList()
        for idx in range(len(insize)):
            self.up_path.append(conv_block(in_channels=insize[idx],
                                           out_channels=outsize[idx]))
        self.fc = nn.Linear(65536, embeddings_dim*self.n_words)

    def forward(self, x):
        for layer in self.up_path:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(x.shape[0]*self.n_words, -1)
        return x
