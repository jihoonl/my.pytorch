import yaml
import os
from .logger import logger


class YAMLLoader(yaml.Loader):

    def __init__(self, stream):
        super(YAMLLoader, self).__init__(stream)
        self._root = os.path.split(stream.name)[0]

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        logger.debug('- Include {}'.format(node.value))
        with open(filename, 'r') as f:
            return yaml.load(f, YAMLLoader)


YAMLLoader.add_constructor('!include', YAMLLoader.include)
