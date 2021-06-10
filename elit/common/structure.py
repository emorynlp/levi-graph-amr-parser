# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 14:58
from typing import Dict

from elit.common.configurable import Configurable
from elit.common.reflection import classpath_of
import json
from collections import OrderedDict
from elit.common.io import filename_is_json, save_pickle, load_pickle, save_json, load_json


class Serializable(object):
    """A super class for save/load operations."""

    def save(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.save_json(path)
            else:
                self.save_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.save_json(path)
        else:
            self.save_pickle(path)

    def load(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.load_json(path)
            else:
                self.load_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.load_json(path)
        else:
            self.load_pickle(path)

    def save_pickle(self, path):
        """Save to path

        Args:
          path:

        Returns:


        """
        save_pickle(self, path)

    def load_pickle(self, path):
        """Load from path

        Args:
          path(str): file path

        Returns:


        """
        item = load_pickle(path)
        return self.copy_from(item)

    def save_json(self, path):
        save_json(self.to_dict(), path)

    def load_json(self, path):
        item = load_json(path)
        return self.copy_from(item)

    # @abstractmethod
    def copy_from(self, item):
        self.__dict__ = item.__dict__
        # raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def to_json(self, ensure_ascii=False, indent=2, sort=False) -> str:
        d = self.to_dict()
        if sort:
            d = OrderedDict(sorted(d.items()))
        return json.dumps(d, ensure_ascii=ensure_ascii, indent=indent, default=lambda o: repr(o))

    def to_dict(self) -> dict:
        return self.__dict__


class SerializableDict(Serializable, dict):

    def save_json(self, path):
        save_json(self, path)

    def copy_from(self, item):
        if isinstance(item, dict):
            self.clear()
            self.update(item)

    def __getattr__(self, key):
        if key.startswith('__'):
            return dict.__getattr__(key)
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def to_dict(self) -> dict:
        return self


class ConfigTracker(Configurable):

    def __init__(self, locals_: Dict, exclude=('kwargs', 'self', '__class__', 'locals_')) -> None:
        """This base class helps sub-classes to capture their arguments passed to ``__init__``, and also their types so
        that they can be deserialized from a config in dict form.

        Args:
            locals_: Obtained by :meth:`locals`.
            exclude: Arguments to be excluded.

        Examples:
            >>> class MyClass(ConfigTracker):
            >>>     def __init__(self, i_need_this='yes') -> None:
            >>>         super().__init__(locals())
            >>> obj = MyClass()
            >>> print(obj.config)
            {'i_need_this': 'yes', 'classpath': 'test_config_tracker.MyClass'}

        """
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        self.config = SerializableDict(
            (k, v.config if hasattr(v, 'config') else v) for k, v in locals_.items() if k not in exclude)
        self.config['classpath'] = classpath_of(self)


class History(object):
    def __init__(self):
        """ A history of training context. It records how many steps have passed and provides methods to decide whether
        an update should be performed, and to caculate number of training steps given dataloader size and
        ``gradient_accumulation``.
        """
        self.num_mini_batches = 0

    def step(self, gradient_accumulation):
        """ Whether the training procedure should perform an update.

        Args:
            gradient_accumulation: Number of batches per update.

        Returns:
            bool: ``True`` to update.
        """
        self.num_mini_batches += 1
        return self.num_mini_batches % gradient_accumulation == 0

    def num_training_steps(self, num_batches, gradient_accumulation):
        """ Caculate number of training steps.

        Args:
            num_batches: Size of dataloader.
            gradient_accumulation: Number of batches per update.

        Returns:

        """
        return len(
            [i for i in range(self.num_mini_batches + 1, self.num_mini_batches + num_batches + 1) if
             i % gradient_accumulation == 0])
