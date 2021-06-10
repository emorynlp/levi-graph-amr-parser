import random

import numpy as np

from elit.common.constant import PAD
from elit.common.vocab import Vocab

DUM, NIL, END = '[unused0]', '<NULL>', '[unused1]'
REL = 'rel='


def list_to_tensor(xs, vocab: Vocab = None, local_vocabs=None, unk_rate=0.):
    pad = vocab.pad_idx if vocab else 0

    def to_idx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [to_idx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.get_idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = to_idx(x, i) + [pad] * (max_len - len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data


def lists_of_string_to_tensor(xs, vocab: Vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD] * (max_len - len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab([DUM] + z + [END]) + [vocab.pad_idx] * (max_string_len - len(z)))
        ys.append(zs)

    data = np.transpose(np.array(ys), (1, 0, 2))
    return data
