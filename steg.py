import os
import glob
import argparse
#
import torch
#
import models
import gif
import definitions as d

parser = argparse.ArgumentParser(description='Load encoder and/or decoder')
parser.add_argument('--n_words', dest='n_words', default=20, type=int,
                    help='Number of words/frame to embed')
parser.add_argument('--deconv', dest='upsample', action='store_false',
                    help='Set model to use deconv instead of upsample blocks')
parser.add_argument('--set_embeddings', dest='embedings_dir',
                    default='./GLoVe', help='Word embeddings location')
parser.add_argument('--set_HIPS', dest='HIPS', default='./HIPS',
                    help='Location to H(idding)Net and R(ecover)Net')
parser.add_argument('--model_flag', dest='flag', default='',
                    help='Set the model identifier flag')
parser.add_argument('--source_fig', dest='source_gif',
                    default='GIF_source', help='Location of source gif')
parser.add_argument('--mode', dest='mode', choices=['E', 'D', 'B'],
                    help='Running mode: encode(E), decode(D) or benchmark(B)')
parser.add_argument('--i_mode', dest='i_mode', default='K', choices=['K', 'S', 'F'],
                    help='Select input mode for secret message \
                    (Keyboard or  File). Only used if string is not passed')
parser.add_argument('--message', action="store", default=None, type=str)
parser.add_argument('--n_images', dest='n_images', default=0, type=int,
                    help='Specifies the number of images to use. Default will\
 use minimum needed.')


def run_encode(encoder_decoder, hnet, rnet, source_gif, message, device,
               validate=True):
    """ Function call to encode messages in an image and hide them
    Parameters
    ----------
    encoder_decoder : StegEncoderDecoder
        Encoder-Decoder imported from models
    hnet : HiddingNet
        Hidding Unet imported from models
    rnet : RecoveryNet
        Recovery network imported from models
    source_gif : str
        Path to source gif to be used for cover
    message : str
        Message to encode and hide
    device : torch.device
        Device used to process the data
    validate : bool
        Swith to turn message validation on/off.  If true will decode the
        message and display
    Returns
    -------
    """
    #
    print('Processing cover frames: {}'.format(source_gif))
    gif.process_gif(source_gif)
    gif_dataset = gif.GIFDataset(folder_path='./gif')
    #
    labels = encoder_decoder.message_2_labels(message)
    embeddings = encoder_decoder.label_2_embeddings(torch.LongTensor(labels)).to(device)
    #
    print('Encoding message')
    gif.encode_gif(gif_dataset=gif_dataset, hider=hnet,
                   encoder_decoder=encoder_decoder, embeddings=embeddings,
                   device=device)
    # clean-up
    files = glob.glob('./gif/*') + glob.glob('./container_gif/*')
    for f in files:
        os.remove(f)
    #
    if validate:
        print('Validating encoding')
        reco_message = gif.decode_gif(reveal_net=rnet,
                                      encoder_decoder=encoder_decoder,
                                      gif_path='./message.png', device=device)
        print('Decoded message: \n{}'.format(reco_message))
        files = glob.glob('./container_gif/*')
        for f in files:
            os.remove(f)


def run_decode(encoder_decoder, rnet, device):
    """ Function call to decode message hidden in container gif. Displays to
    screen and saves to file.
    Parameters
    ----------
    encoder_decoder : StegEncoderDecoder
        Encoder-Decoder imported from models
    rnet : RecoveryNet
        Recovery network imported from model
    device : torch.device
        Device to run the processes on

    Returns
    -------
    """
    message_gif = input('Input gif file: ')
    print('Decoding message...')
    reco_message = gif.decode_gif(reveal_net=rnet,
                                  encoder_decoder=encoder_decoder,
                                  gif_path=message_gif, device=device)
    files = glob.glob('./gif/*')
    for f in files:
        os.remove(f)
    print('Decoded message: \n{}'.format(reco_message))
    with open("./message.txt", "w") as text_file:
        text_file.write(reco_message)  


if __name__ == '__main__':
    args = parser.parse_args()
    tag = ''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading models ...')
    encoder_decoder = models.StegEncoderDecoder(embedding_path=d.embeddings,
                                                n_words=args.n_words,
                                                encoder_path=d.encoder,
                                                decoder_path=d.decoder,
                                                device=device)
    rnet = models.Rnet(Rnet_path=d.rnet, device=device)
    hnet = models.Hnet(Hnet_path=d.hnet, device=device)
    #
    if args.mode == 'E':
        if args.message is not None:
            message = args.message.lower()
        elif args.i_mode == 'K':
            message = input('Input secret message: ').lower()
        else:
            message_file = input('Input secret message: ')
            with open(message_file) as file:
                message = file.read().replace('\n', '').lower()
        #
        if args.upsample:
            tag = 'upsample'
        run_encode(encoder_decoder=encoder_decoder, hnet=hnet, rnet=rnet,
                   source_gif=args.source_gif, message=message, device=device)

    elif args.mode == 'D':
        run_decode(encoder_decoder=encoder_decoder, rnet=rnet, device=device)

    elif args.mode == 'B':
        print('Benchmark, still note implemented in this function')
