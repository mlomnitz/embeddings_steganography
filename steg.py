import argparse
#
import torch
#
from models import encoder, decoder
from glove_aec import encoder_decoder
from definitions import local_path

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
parser.add_argument('--i_mode', dest='i_mode', default='K', choices=['K', 'F'],
                    help='Select input mode for secret message \
(Keyboard or File)')
parser.add_argument('--n_images', dest='n_images', default=0, type=int,
                    help='Specifies the number of images to use. Default will\
 use minimum needed.')


def run_encode(n_words, n_images, tag, input_mode):
    #
    if input_mode == 'K':
        message = input('Input secret message: ').lower()
    else:
        message_file = input('Input secret message: ')
        with open(message_file) as file:
            message = file.read().replace('\n', '').lower()
    #
    generator = encoder(n_words=args.n_words, upsample=args.upsample)
    chpt = torch.load('{}/weights/embedding_{}_word_encoder_{}.pth.tar'
                      .format(local_path, n_words, tag))
    generator.load_state_dict(chpt['state_dict'])
    #
    classifier = decoder(n_words=args.n_words)
    chpt = torch.load('{}/weights/embedding_{}_word_decoder_{}.pth.tar'
                      .format(local_path, n_words, tag))
    classifier.load_state_dict(chpt['state_dict'])
    #
    aec = encoder_decoder(encoder=generator, decoder=classifier)
    aec.encode_message(message=message, n_images=n_images)
    aec.decode_message(gif='./Example/message.png', save_path=None)
    print('Decoded message: \n{}'.format(message))


def run_decode(n_words, tag):
    message_gif = input('Input gif file: ')
    classifier = decoder(n_words=args.n_words)
    chpt = torch.load('{}/weights/embedding_{}_word_decoder_{}.pth.tar'
                      .format(local_path, n_words, tag))
    classifier.load_state_dict(chpt['state_dict'])
    aec = encoder_decoder(decoder=classifier)
    message = aec.decode_message(gif=message_gif,
                                 save_path='./Example/message.txt')
    print('Decoded message: \n{}'.format(message))


if __name__ == '__main__':
    args = parser.parse_args()
    tag = ''
    if args.upsample:
        tag = 'upsample'
    if args.mode == 'E':
        run_encode(args.n_words, args.n_images, tag, args.i_mode)

    elif args.mode == 'D':
        run_decode(args.n_words, tag)

    elif args.mode == 'B':
        print('Benchmark, still note implemented in this function')
