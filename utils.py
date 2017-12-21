from collections import defaultdict

import numpy as np

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

def calculate_perplexity(log_probs):
    # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    perp = 0
    for p in log_probs:
        perp += -p
    return np.exp(perp / len(log_probs))

def get_ptb_dataset(dataset='train'):
    fn = 'data/ptb/ptb.{}.txt'
    for line in open(fn.format(dataset)):
        for word in line.split():
            yield word
        # Add token to the end of the line
        # Equivalent to <eos> in:
        # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
        yield '<eos>'

def ptb_iterator(raw_data, batch_size, num_steps):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
    # Optionally shuffle the data before training (预处理：index随机乱序)
    if shuffle:
        # 返回一个随机排列（0-len(orig_X)）
        '''
        如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；而shuffle只是对一个矩阵进行洗牌，无返回值。
        >>> np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12])
        
        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.permutation(arr)
        array([[6, 7, 8],
              [0, 1, 2],
              [3, 4, 5]])
        -------------------------------
        import numpy as np

        a = np.array([[1,2], [3, 4], [5, 6]])
        print(a[2,0,1]) # output: [[5,6],[1,2],[3,4]]

        # An example of integer array indexing.
        # The returned array will have shape (3,) and 
        print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"

        # The above example of integer array indexing is equivalent to this:
        print np.array([a[0, 0], a[1, 1], a[2, 0]])  # Prints "[1 4 5]"
        '''
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if np.any(orig_y) else None
    else:
        data_X = orig_X
        data_y = orig_y
    # ##
    total_processed_examples = 0
    # np.ceil(n) 返回不小于 n 的最小整数 total_steps：相当于跑一个epoch需要的迭代次数
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in range(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        # Convert our target from the class index to a one hot vector (转换成 one-hot 形式)
        y = None
        if np.any(data_y):
            y_indices = data_y[batch_start:batch_start + batch_size]
            y = np.zeros((len(x), label_size), dtype=np.int32)
            y[np.arange(len(y_indices)), y_indices] = 1
        # ##
        yield x, y
        total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
