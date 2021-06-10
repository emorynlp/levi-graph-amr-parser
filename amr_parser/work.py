import torch
import os

from elit.components.amr.amr_parser.data import REL
from elit.datasets.parsing.amr import unlinearize, remove_unconnected_components, un_kahn
from elit.transform.transformer_tokenizer import TransformerSequenceTokenizer

if os.environ.get('USE_TF', None) is None:
    os.environ["USE_TF"] = 'NO'  # saves time loading transformers
from amr_parser.data import Vocab, DataLoader, DUM, END, CLS, NIL, seperate_concept_from_rel
from amr_parser.parser import Parser
from amr_parser.postprocess import PostProcessor
from amr_parser.extract import LexicalMap
from amr_parser.utils import move_to_device
from amr_parser.bert_utils import BertEncoderTokenizer, BertEncoder, load_bert
from amr_parser.match import match

import argparse, os, re


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--max_time_step', type=int)
    parser.add_argument('--output_suffix', type=str)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()


def show_progress(model, dev_data):
    model.eval()
    loss_acm = 0.
    for batch in dev_data:
        batch = move_to_device(batch, model.device)
        concept_loss, arc_loss, rel_loss = model(batch)
        loss = concept_loss + arc_loss + rel_loss
        loss_acm += loss.item()
    print('total loss', loss_acm)
    return loss_acm


def parse_batch(model, batch, beam_size, alpha, max_time_step):
    levi_graph = model.decoder.levi_graph
    device = model.device
    res = dict()
    concept_batch = []
    relation_batch = []
    beams = model.work(batch, beam_size, max_time_step)
    score_batch = []
    idx = 0
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_concept = [token for token in best_hyp.seq[1:-1]]
        predicted_rel = []
        if levi_graph:
            last_concept_id = -1
            edge = []
            rel_mask = [x.startswith(REL) for x in predicted_concept]
            rel_mask = torch.tensor(rel_mask, dtype=torch.bool, device=device)
            for i, c in enumerate(predicted_concept):
                if i == 0:
                    if not c.startswith(REL):
                        last_concept_id = i
                    continue
                if levi_graph is True:
                    if c.startswith(REL):
                        # At least two concepts are needed to make a relation, so id >= 1
                        if last_concept_id >= 1:
                            arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]
                            arc[rel_mask[:i]] = 0
                            v = last_concept_id
                            p, u = arc[:v].max(0)
                            u = int(u)
                            edge.append((v, i, ''))
                            edge.append((i, u, ''))
                    else:
                        last_concept_id = i
                else:
                    if c.startswith(REL):
                        mask = rel_mask
                    else:
                        mask = ~rel_mask
                    arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]
                    arc[mask[:i]] = 0
                    if c.startswith(REL):
                        p, u = arc[:i].max(0)
                        p = float(p)
                        u = int(u)
                        edge.append((u, i, p))
                    else:
                        # concept can have multiple heads
                        for u, p in enumerate(arc[:i].tolist()):
                            p = float(p)
                            if p < 0.5:
                                continue
                            edge.append((u, i, p))
            if levi_graph is True:
                c, e = unlinearize(predicted_concept, edge)
            else:
                c, e = un_kahn(predicted_concept, edge)
            # Prune unconnected concept
            # c, e = remove_unconnected_components(c, e)
            predicted_concept = c
            predicted_rel = e
        else:
            for i in range(len(predicted_concept)):
                if i == 0:
                    continue
                arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]  # head_len
                rel = best_hyp.state_dict['rel_ll%d' % i].squeeze_().exp_()[1:, :]  # head_len x vocab
                for head_id, (arc_prob, rel_prob) in enumerate(zip(arc.tolist(), rel.tolist())):
                    predicted_rel.append((i, head_id, arc_prob, rel_prob))
        concept_batch.append(predicted_concept)
        score_batch.append(best_hyp.score)
        relation_batch.append(predicted_rel)
        idx += 1
    res['concept'] = concept_batch
    res['score'] = score_batch
    res['relation'] = relation_batch
    return res


def parse_data(model, pp, data, input_file, output_file, beam_size=8, alpha=0.6, max_time_step=100):
    tot = 0
    levi_graph = model.decoder.levi_graph
    with open(output_file, 'w') as fo:
        for batch in data:
            batch = move_to_device(batch, model.device)
            res = parse_batch(model, batch, beam_size, alpha, max_time_step)
            for concept, relation, score in zip(res['concept'], res['relation'], res['score']):
                fo.write('# ::conc ' + ' '.join(concept) + '\n')
                fo.write('# ::score %.6f\n' % score)
                fo.write(pp.postprocess(concept, relation, levi_graph) + '\n\n')
                tot += 1
    match(output_file, input_file)
    # print ('write down %d amrs'%tot)


def load_ckpt_without_bert(model, test_model):
    ckpt = torch.load(test_model, map_location=torch.device('cpu'))['model']
    for k, v in model.state_dict().items():
        if k.startswith('bert_encoder'):
            ckpt[k] = v
    model.load_state_dict(ckpt, strict=False)


def main():
    args = parse_config()
    load_path = args.load_path
    device = args.device
    output_suffix = args.output_suffix
    beam_size = args.beam_size
    max_time_step = args.max_time_step
    alpha = args.alpha
    test_data = args.test_data
    test_batch_size = args.test_batch_size

    predict(load_path, test_data, alpha, beam_size, device, max_time_step, output_suffix, test_batch_size)


def predict(load_path, test_data, alpha=0.6, beam_size=8, device=0, max_time_step=100, output_suffix='_test_out',
            test_batch_size=6666):
    test_models = []
    if os.path.isdir(load_path):
        for file in os.listdir(load_path):
            fname = os.path.join(load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']
    else:
        test_models.append(load_path)
        model_args = torch.load(load_path, map_location=torch.device('cpu'))['args']

    for k, v in vars(model_args).items():
        if k.endswith('_vocab'):
            model_vocab = os.path.join(os.path.dirname(load_path), os.path.basename(v))
            if os.path.isfile(model_vocab):
                setattr(model_args, k, model_vocab)

    vocabs = dict()
    vocabs['tok'] = Vocab(model_args.tok_vocab, 5, [CLS])
    vocabs['lem'] = Vocab(model_args.lem_vocab, 5, [CLS])
    vocabs['pos'] = Vocab(model_args.pos_vocab, 5, [CLS])
    vocabs['ner'] = Vocab(model_args.ner_vocab, 5, [CLS])
    vocabs['predictable_concept'] = Vocab(model_args.predictable_concept_vocab, 5, [DUM, END])
    vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [DUM, END])
    vocabs['rel'] = Vocab(model_args.rel_vocab, 50, [NIL])
    vocabs['word_char'] = Vocab(model_args.word_char_vocab, 100, [CLS, END])
    vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [CLS, END])
    if hasattr(model_args, 'separate_rel') and model_args.separate_rel:
        seperate_concept_from_rel(vocabs)
    lexical_mapping = LexicalMap()
    bert_encoder = None
    if model_args.with_bert:
        bert_path = model_args.bert_path
        if 'bert-base-cased' in bert_path:
            bert_path = 'bert-base-cased'
        bert_tokenizer = BertEncoderTokenizer.from_pretrained(bert_path, do_lower_case=False)
        # tokenizer = TransformerSequenceTokenizer(model_args.bert_path, 'token', use_fast=False, do_basic_tokenize=False,
        #                                          cls_is_bos=True)
        bert_encoder = load_bert(bert_path)
        vocabs['bert_tokenizer'] = bert_tokenizer
    if device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', device)
    levi_graph = model_args.levi_graph if hasattr(model_args, 'levi_graph') else False
    model = Parser(vocabs,
                   model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,
                   model_args.concept_char_dim, model_args.concept_dim,
                   model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
                   model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
                   model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
                   bert_encoder=bert_encoder, device=device,
                   joint_arc_concept=hasattr(model_args, 'joint_arc_concept') and model_args.joint_arc_concept,
                   levi_graph=levi_graph)
    another_test_data = DataLoader(vocabs, lexical_mapping, test_data, test_batch_size, for_train=False,
                                   levi_graph=levi_graph)
    for test_model in test_models:
        print(test_model)

        load_ckpt_without_bert(model, test_model)
        model = model.to(device)
        model.eval()

        # loss = show_progress(model, test_data)
        pp = PostProcessor(vocabs['rel'])
        parse_data(model, pp, another_test_data, test_data, test_model + output_suffix, beam_size, alpha, max_time_step)


if __name__ == "__main__":
    main()
