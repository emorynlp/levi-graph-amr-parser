# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 19:24
import os
import traceback
from sys import exit

from elit.common.constant import HANLP_VERBOSE

from elit import pretrained
from elit.common.component import Component
from elit.utils.io_util import get_resource, get_latest_info_from_pypi
from elit.common.io import load_json, eprint
from elit.common.reflection import object_from_classpath, str_to_type
from elit import version


def load_from_meta_file(save_dir: str, meta_filename='meta.json', transform_only=False, verbose=HANLP_VERBOSE,
                        **kwargs) -> Component:
    """

    Args:
        save_dir:
        meta_filename (str): The meta file of that saved component, which stores the classpath and version.
        transform_only:
        **kwargs:

    Returns:

    """
    identifier = save_dir
    load_path = save_dir
    save_dir = get_resource(save_dir)
    if save_dir.endswith('.json'):
        meta_filename = os.path.basename(save_dir)
        save_dir = os.path.dirname(save_dir)
    metapath = os.path.join(save_dir, meta_filename)
    if not os.path.isfile(metapath):
        tf_model = False
        metapath = os.path.join(save_dir, 'config.json')
    else:
        tf_model = True
    if not os.path.isfile(metapath):
        tips = ''
        if save_dir.isupper():
            from difflib import SequenceMatcher
            similar_keys = sorted(pretrained.ALL.keys(),
                                  key=lambda k: SequenceMatcher(None, save_dir, metapath).ratio(),
                                  reverse=True)[:5]
            tips = f'Check its spelling based on the available keys:\n' + \
                   f'{sorted(pretrained.ALL.keys())}\n' + \
                   f'Tips: it might be one of {similar_keys}'
        raise FileNotFoundError(f'The identifier {save_dir} resolves to a non-exist meta file {metapath}. {tips}')
    meta: dict = load_json(metapath)
    cls = meta.get('classpath', None)
    if not cls:
        cls = meta.get('class_path', None)  # For older version
    if tf_model:
        # tf models are trained with version <= 2.0. To migrate them to 2.1, map their classpath to new locations
        upgrade = {
            'elit.components.tok.TransformerTokenizer': 'elit.components.tok_tf.TransformerTokenizerTF',
            'elit.components.pos.RNNPartOfSpeechTagger': 'elit.components.pos_tf.RNNPartOfSpeechTaggerTF',
            'elit.components.pos.CNNPartOfSpeechTagger': 'elit.components.pos_tf.CNNPartOfSpeechTaggerTF',
            'elit.components.ner.TransformerNamedEntityRecognizer': 'elit.components.ner_tf.TransformerNamedEntityRecognizerTF',
            'elit.components.parsers.biaffine_parser.BiaffineDependencyParser': 'elit.components.parsers.biaffine_parser_tf.BiaffineDependencyParserTF',
            'elit.components.parsers.biaffine_parser.BiaffineSemanticDependencyParser': 'elit.components.parsers.biaffine_parser_tf.BiaffineSemanticDependencyParserTF',
            'elit.components.tok.NgramConvTokenizer': 'elit.components.tok_tf.NgramConvTokenizerTF',
            'elit.components.classifiers.transformer_classifier.TransformerClassifier': 'elit.components.classifiers.transformer_classifier_tf.TransformerClassifierTF',
            'elit.components.taggers.transformers.transformer_tagger.TransformerTagger': 'elit.components.taggers.transformers.transformer_tagger_tf.TransformerTaggerTF',
        }
        cls = upgrade.get(cls, cls)
    assert cls, f'{meta_filename} doesn\'t contain classpath field'
    try:
        obj: Component = object_from_classpath(cls)
        if hasattr(obj, 'load'):
            if transform_only:
                # noinspection PyUnresolvedReferences
                obj.load_transform(save_dir)
            else:
                if os.path.isfile(os.path.join(save_dir, 'config.json')):
                    obj.load(save_dir, verbose=verbose, **kwargs)
                else:
                    obj.load(metapath, **kwargs)
            obj.config['load_path'] = load_path
        return obj
    except Exception as e:
        eprint(f'Failed to load {identifier}. See traceback below:')
        eprint(f'{"ERROR LOG BEGINS":=^80}')
        traceback.print_exc()
        eprint(f'{"ERROR LOG ENDS":=^80}')
        if isinstance(e, ModuleNotFoundError):
            eprint('Some modules required by this model are missing. Please install the full version:\n'
                   'pip install elit[full]')
        from pkg_resources import parse_version
        model_version = meta.get("hanlp_version", "unknown")
        if model_version == '2.0.0':  # Quick fix: the first version used a wrong string
            model_version = '2.0.0-alpha.0'
        model_version = parse_version(model_version)
        installed_version = parse_version(version.__version__)
        try:
            latest_version = get_latest_info_from_pypi()
        except:
            latest_version = None
        if model_version > installed_version:
            eprint(f'{identifier} was created with elit-{model_version}, '
                   f'while you are running a lower version: {installed_version}. ')
        if installed_version != latest_version:
            eprint(
                f'Please upgrade elit with:\n'
                f'pip install --upgrade elit\n')
        eprint(
            'If the problem still persists, please submit an issue to https://github.com/hankcs/ELIT/issues\n'
            'When reporting an issue, make sure to paste the FULL ERROR LOG above.')
        exit(1)


def load_from_meta(meta: dict) -> Component:
    if 'load_path' in meta:
        return load_from_meta_file(meta['load_path'])
    cls = meta.get('class_path', None) or meta.get('classpath', None)
    assert cls, f'{meta} doesn\'t contain classpath field'
    cls = str_to_type(cls)
    return cls.from_config(meta)
