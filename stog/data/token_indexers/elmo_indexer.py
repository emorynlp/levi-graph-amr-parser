from typing import Dict, List

from overrides import overrides

from stog.utils.checks import ConfigurationError
from stog.utils.string import pad_sequence_to_length
from stog.data.tokenizers.token import Token
from stog.data.token_indexers.token_indexer import TokenIndexer
from stog.data.vocabulary import Vocabulary


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids

class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = '<S>'
    eos_token = '</S>'

    @staticmethod
    def convert_word_to_char_ids(word: str) -> List[int]:
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]


@TokenIndexer.register("elmo_characters")
class ELMoTokenCharactersIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'elmo_characters') -> None:
        self._namespace = namespace


    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass


    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument
        texts = [token.text for token in tokens]

        if any(text is None for text in texts):
            raise ConfigurationError('ELMoTokenCharactersIndexer needs a tokenizer '
                                     'that retains text')
        return {index_name: [ELMoCharacterMapper.convert_word_to_char_ids(text) for text in texts]}


    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}


    def get_padding_token(self) -> List[int]:
        return []

    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length


    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key],
                                            default_value=self._default_value_for_padding)
                for key, val in tokens.items()}
