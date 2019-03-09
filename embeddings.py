import  numpy as np
import bcolz
import pickle
from tqdm import tqdm


def load_glove_file(glove_path):

    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/6B.50.dat'.format(glove_path),
                           mode='w')
    
    total_lines = sum(1 for line in open('{}/glove.6B.50d.txt'.format(glove_path),
                                         'rb'))
    with open('{}/glove.6B.50d.txt'.format(glove_path), 'rb') as f:
        progress_bar = tqdm(total=total_lines)
        for l in f:
            progress_bar.update(1)
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    progress_bar.close()
    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir='{}/6B.50.dat'.format(glove_path), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/6B.50_words.pkl'.format(glove_path), 'wb'))
    pickle.dump(word2idx, open('{}/6B.50_idx.pkl'.format(glove_path), 'wb'))


class glove():
    def __init__(self, glove_path='../GLoVE'):
        self.vectors = bcolz.open('{}/6B.50.dat'.format(glove_path))[:]
        self.words = pickle.load(open('{}/6B.50_words.pkl'.format(glove_path), 'rb'))
        self.word2idx = pickle.load(open('{}/6B.50_idx.pkl'.format(glove_path), 'rb'))
        self.words[self.word2idx['blank']] = ''
        self.vocab_size = len(self.vectors)
    
    
