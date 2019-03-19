import random
import sys
import nltk
import itertools
from collections import defaultdict
from pyvi import ViTokenizer
import numpy as np
import pickle
from random import sample

FILENAME = 'data/data.txt'

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 1
        }

UNK = 'unk'
# VOCAB_SIZE = 10000
VOCAB_SIZE = 3900

def ddefault():
    return 1
    
# Đọc file

def read_lines(filename):
    return open(filename, encoding='utf-8', errors='ignore').read().split('\n')[:-1]

'''
	- Các từ thông dụng
	- Tập index2word
	- Tập word2index
'''

def index_(tokenized_sentences, vocab_size):
    # Danh sách các từ thường sử dụng
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print('len freq_dist: {}'.format(len(freq_dist)))
    # Danh sách các 10000 từ thường sử dụng nhất ex:(abc, 50)
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0].replace('_', ' ') for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

# Lọc ra những câu quá dài hoặc quá ngắn

# Lọc ra những câu quá dài hoặc quá ngắn

def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    i = 0
    while (i < len(sequences)-1):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(ViTokenizer.tokenize(sequences[i]))
                filtered_a.append(ViTokenizer.tokenize(sequences[i+1]))
        i = i + 3

    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print("len of filtered data: {}".format(filt_data_len*2))
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
  - Chuyển danh sách các items sang mảng các chỉ số
  - Thêm padding
'''
def zero_pad(qtokenized, atokenized, w2idx):
    for i in range(len(qtokenized)):
        qtokenized[i] = [j.replace('_', ' ') for j in qtokenized[i]]
    for i in range(len(atokenized)):
        atokenized[i] = [j.replace('_', ' ') for j in atokenized[i]]
    data_len = len(qtokenized)
    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)
  
    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print("--------")
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

'''
	Thay thế các từ không biết (không nằm trong w2idx) bằng kí tự UNK
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # Chuyển qua chữ thường
    lines = [ line.lower() for line in lines ]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    # Lọc ra nhưng câu quá ngắn hoặc quá dài
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)

    # Cắt từ trong câu
    print('\n>> Segment lines into words')
    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]

    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # Save
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH=''):
    try:
        with open(PATH + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None
        
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

# Chia data thành 70% train, 15% test, 15% xác thực

def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)

# Tạo batch từ tập dataset

def batch_gen(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T
                
# Tạo batch bằng cách lấy ngẫu nhiên 1 tập items

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

# decode

def decode(sequence, lookup, separator=''): # 0 được sử dụng để padding
    return separator.join([ lookup[element] for element in sequence if element ])

if __name__ == '__main__':
    process_data()
