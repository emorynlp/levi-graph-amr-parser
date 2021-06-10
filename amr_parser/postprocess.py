import re
from collections import defaultdict

import numpy as np
import penman

from amr_parser.AMRGraph import is_attr_or_abs_form, need_an_instance
from amr_parser.data import Vocab
from elit.datasets.parsing.amr import largest_connected_component, to_triples


class PostProcessor(object):
    def __init__(self, rel_vocab):
        self.amr = penman.AMRCodec()
        self.rel_vocab: Vocab = rel_vocab

    def to_triple(self, res_concept, res_relation):
        """
        Take argmax of relations as prediction when prob of arc >= 0.5 (sigmoid).

        Args:
          res_concept: list of strings
          res_relation: list of (dep:int, head:int, arc_prob:float, rel_prob:list(vocab))

        Returns:

        """
        ret = []
        names = []
        for i, c in enumerate(res_concept):
            if need_an_instance(c):
                name = 'c' + str(i)
                ret.append((name, 'instance', c))
            else:
                if c.endswith('_'):
                    name = '"' + c[:-1] + '"'
                else:
                    name = c
                name = name + '@attr%d@' % i
            names.append(name)

        grouped_relation = defaultdict(list)
        # dep head arc rel
        for T in res_relation:
            if len(T) == 4:
                i, j, p, r = T
                r = self.rel_vocab.get_token(int(np.argmax(np.array(r))))
            else:
                i, j, r = T
                p = 1
            grouped_relation[i] = grouped_relation[i] + [(j, p, r)]
        for i, c in enumerate(res_concept):
            if i == 0:
                continue
            max_p, max_j, max_r = 0., 0, None
            for j, p, r in grouped_relation[i]:
                assert j < i
                if is_attr_or_abs_form(res_concept[j]):
                    continue
                if p >= 0.5:
                    if not is_attr_or_abs_form(res_concept[i]):
                        if r.endswith('_reverse_'):
                            ret.append((names[i], r[:-9], names[j]))
                        else:
                            ret.append((names[j], r, names[i]))
                if p > max_p:
                    max_p = p
                    max_j = j
                    max_r = r
            if not max_r:
                continue
            if max_p < 0.5 or is_attr_or_abs_form(res_concept[i]):
                if max_r.endswith('_reverse_'):
                    ret.append((names[i], max_r[:-9], names[max_j]))
                else:
                    ret.append((names[max_j], max_r, names[i]))
        return ret

    def get_string(self, x):
        return self.amr.encode(penman.Graph(x), top=x[0][0])

    def postprocess(self, concept, relation, check_connected=False):
        triples = self.to_triple(concept, relation)
        if check_connected:
            c, e = largest_connected_component(triples)
            triples = to_triples(c, e)
        if check_connected:
            mstr = None
            while not mstr:
                try:
                    if not triples:
                        mstr = '(c0 / multi-sentence)'
                    else:
                        mstr = self.get_string(triples)
                except penman.EncodeError:
                    triples = triples[:-1]
        else:
            mstr = self.get_string(triples)
        return re.sub(r'@attr\d+@', '', mstr)
