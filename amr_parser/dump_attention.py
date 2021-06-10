import torch
import os

from elit.components.amr.amr_parser.data import REL
from elit.datasets.parsing.amr import unlinearize, remove_unconnected_components, un_kahn

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

    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
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


def load_ckpt_without_bert(model, test_model):
    ckpt = torch.load(test_model, map_location=torch.device('cpu'))['model']
    for k, v in model.state_dict().items():
        if k.startswith('bert_encoder'):
            ckpt[k] = v
    model.load_state_dict(ckpt, strict=False)


def predict(load_path, test_data, device=0, test_batch_size=6666):
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
    # if 'joint' in test_data or 'joint' in load_path:
    #     model_args.tok_vocab = '/home/hhe43/elit/data/amr/amr_2.0/tok_vocab'
    #     model_args.lem_vocab = '/home/hhe43/elit/data/amr/amr_2.0/lem_vocab'
    #     model_args.pos_vocab = '/home/hhe43/elit/data/amr/amr_2.0/pos_vocab'
    #     model_args.ner_vocab = '/home/hhe43/elit/data/amr/amr_2.0/ner_vocab'
    #     model_args.predictable_concept_vocab = '/home/hhe43/elit/data/amr/amr_2.0/predictable_concept_vocab'
    #     model_args.concept_vocab = '/home/hhe43/elit/data/amr/amr_2.0/concept_vocab'
    #     model_args.rel_vocab = '/home/hhe43/elit/data/amr/amr_2.0/rel_vocab'
    #     model_args.word_char_vocab = '/home/hhe43/elit/data/amr/amr_2.0/word_char_vocab'
    #     model_args.concept_char_vocab = '/home/hhe43/elit/data/amr/amr_2.0/concept_char_vocab'

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
                   joint_arc_concept=hasattr(model_args, 'model_args') and model_args.joint_arc_concept,
                   levi_graph=levi_graph)
    test_data = DataLoader(vocabs, lexical_mapping, test_data, test_batch_size, for_train=False,
                           levi_graph=levi_graph)
    for test_model in test_models:
        load_ckpt_without_bert(model, test_model)
        model = model.to(device)
        model.train()
        for batch in test_data:
            batch = move_to_device(batch, model.device)
            concept_loss, arc_loss, rel_loss, graph_arc_loss = model(batch)
            break


def main():
    args = parse_config()
    load_path = args.load_path
    device = args.device

    predict(load_path, args.test_data, device)


if __name__ == "__main__":
    main()
