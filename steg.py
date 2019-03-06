import argparse
#
import torch
#
from models import encoder, decoder
from glove_aec import encoder_decoder
from definitions import local_path

parser = argparse.ArgumentParser(description='Load encoder and/or decoder')
parser.add_argument('--n_words', dest='n_words', default=5, type=int,
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


def run_encode(n_words, tag):
    message_file = input('Input message file: ')
    with open(message_file) as file:
        message = file.read().replace('\n', '').lower()
    #
    generator = encoder(n_words=args.n_words, upsample=args.upsample)
    chpt = torch.load('{}/weights/embedding_{}_word_encoder_{}.pth.tar'
                      .format(local_path, n_words, tag))
    generator.load_state_dict(chpt['state_dict'])
    aec = encoder_decoder(encoder=generator)
    aec.encode_message(message=message)


def run_decode(n_words, tag):
    message_gif = input('Input gif file: ')
    classifier = decoder(n_words=args.n_words)
    chpt = torch.load('{}/weights/embedding_{}_word_decoder_{}.pth.tar'
                      .format(local_path, n_words, tag))
    classifier.load_state_dict(chpt['state_dict'])
    aec = encoder_decoder(decoder=classifier)
    aec.decode_message(gif=message_gif, save_path='./Example/message.txt')

    
if __name__ == '__main__':
    args = parser.parse_args()
    tag = ''
    if args.upsample:
        tag = 'upsample'
    if args.mode == 'E':
        run_encode(args.n_words, tag)

    elif args.mode == 'D':
        run_decode(args.n_words, tag)
        
    elif args.mode == 'B':
        print('Benchmark, still note implemented in this function')
        
    
