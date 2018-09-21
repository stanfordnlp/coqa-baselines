import os
import json
import sys
from datetime import datetime
from .graph_utils import plot_learn, plot_metrics
from .data_utils import get_file_contents
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

    def __init__(self, config, dirname=None, saved_dir=None):
        self.config = config
        self.dirname = saved_dir.split("/")[-1] if saved_dir else self.make_directory(dirname)
        sys.stdout = Logger(os.path.join(Constants._RESULTS_DIR, self.dirname, Constants._LOG_FILE))
        if saved_dir is None:
            self.log_json(self.config, os.path.join(Constants._RESULTS_DIR, self.dirname, Constants._CONFIG_FILE))

    def make_directory(self, dirname=None):
        if not dirname:
            timestamp = str(datetime.now().isoformat().replace(':', '-').replace('T', '_')[:-7])
            dirname = '_'.join(["model", timestamp])

        full_dirname = '{}/{}'.format(Constants._RESULTS_DIR, dirname)
        if os.path.exists(full_dirname):
            raise Exception('Directory already exists: {}'.format(full_dirname))
        os.mkdir(full_dirname)
        os.mkdir('{}/{}'.format(full_dirname, "metrics"))
        os.mkdir('{}/{}'.format(full_dirname, "graphs"))
        return dirname

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
        if not os.path.isdir(os.path.join(Constants._RESULTS_DIR, self.dirname)):
            raise NameError('Error: %s model directory not found' % self.dirname)

        if isinstance(data, list):
            data = '\n'.join([str(i) for i in data])

        path = os.path.join(Constants._RESULTS_DIR, self.dirname, filename)
        with open(path, 'a') as f:
            f.write('%s\n' % data)
