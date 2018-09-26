import os
import json
import sys
from . import constants as Constants


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class ModelLogger(object):

    def __init__(self, config, dirname=None):
        self.config = config
        self.dirname = dirname
        if os.path.exists(dirname):
            raise Exception('Directory already exists: {}'.format(dirname))
        os.mkdir(dirname)
        os.mkdir('{}/{}'.format(dirname, "metrics"))
        sys.stdout = Logger(os.path.join(dirname, Constants._LOG_FILE))
        self.log_json(self.config, os.path.join(self.dirname, Constants._CONFIG_FILE))

    def log_json(self, data, filename, mode='w'):
        with open(filename, mode) as outfile:
            outfile.write(json.dumps(data, indent=4, ensure_ascii=False))

    def log(self, data, filename):
        """Appends the specified data in plaintext to the logging file.
        Args:
        1. data as anything (the data to write)
        2. filename as string (the particular file within that directory this data
        should go in)
        """
        if not os.path.isdir(self.dirname):
            raise NameError('Error: %s model directory not found' % self.dirname)

        if isinstance(data, list):
            data = '\n'.join([str(i) for i in data])

        path = os.path.join(self.dirname, filename)
        with open(path, 'a') as f:
            f.write('%s\n' % data)
