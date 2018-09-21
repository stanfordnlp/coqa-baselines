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

    def graph_learning_curves(self):
        """
        Used at the end of a training cycle to populate the 'graphs' dir with all
        6 learning curves: epoch and iteration loss/f1/em. Assumes all log files are
        fully populated.
        """
        dirpath = os.path.join(Constants._RESULTS_DIR, self.dirname)
        graphdir = os.path.join(dirpath, "graphs")
        modelType = self.config["model"]
        if not os.path.isdir(dirpath):
            raise NameError('Error: %s model directory not found' % dirpath)

        # Grab all plottable log data
        epoch_train_span_loss = get_file_contents(os.path.join(dirpath, Constants._TRAIN_SPAN_LOSS_EPOCH_LOG)).split()
        epoch_train_chunk_loss = get_file_contents(os.path.join(dirpath, Constants._TRAIN_CHUNK_LOSS_EPOCH_LOG)).split()
        epoch_train_f1 = get_file_contents(os.path.join(dirpath, Constants._TRAIN_F1_EPOCH_LOG)).split()
        epoch_train_em = get_file_contents(os.path.join(dirpath, Constants._TRAIN_EM_EPOCH_LOG)).split()
        if self.config["model"] == "hdrqa":
            epoch_train_acc = get_file_contents(os.path.join(dirpath, Constants._TRAIN_CHUNK_ANY_ACC_EPOCH_LOG)).split()

        # epoch_dev_loss = get_file_contents(os.path.join(dirpath, Constants._DEV_LOSS_EPOCH_LOG)).split()
        epoch_dev_f1 = get_file_contents(os.path.join(dirpath, Constants._DEV_F1_EPOCH_LOG)).split()
        epoch_dev_em = get_file_contents(os.path.join(dirpath, Constants._DEV_EM_EPOCH_LOG)).split()

        # iter_train_loss = get_file_contents(os.path.join(dirpath, Constants._TRAIN_LOSS_ITER_LOG)).split()
        # iter_train_f1 = get_file_contents(os.path.join(dirpath, Constants._TRAIN_F1_ITER_LOG)).split()
        # iter_train_em = get_file_contents(os.path.join(dirpath, Constants._TRAIN_EM_ITER_LOG)).split()

        # iter_dev_loss = get_file_contents(os.path.join(dirpath, Constants._DEV_LOSS_ITER_LOG)).split()
        # iter_dev_f1 = get_file_contents(os.path.join(dirpath, Constants._DEV_F1_ITER_LOG)).split()
        # iter_dev_em = get_file_contents(os.path.join(dirpath, Constants._DEV_EM_ITER_LOG)).split()

        # Package plot data into 6 train/val graphs
        epoch_span_loss = {self.config["model"]: (epoch_train_span_loss, None)}
        epoch_chunk_loss = {self.config["model"]: (epoch_train_chunk_loss, None)}
        epoch_f1 = {self.config["model"]: (epoch_train_f1, epoch_dev_f1)}
        epoch_em = {self.config["model"]: (epoch_train_em, epoch_dev_em)}

        epoch_vals = [(epoch_train_f1, epoch_dev_f1, "F1"), (epoch_train_em, epoch_dev_em, "EM")]
        if self.config["model"] == "hdrqa":
            epoch_vals += [(epoch_train_acc, None, "Acc")]

        # iter_loss = {self.config["model"]: (iter_train_loss, None)}
        # iter_f1 = {self.config["model"]: (iter_train_f1, iter_dev_f1)}
        # iter_em = {self.config["model"]: (iter_train_em, iter_dev_em)}

        # Plot graphs
        plot_learn(epoch_span_loss, "Loss", "Epochs", title="{} Epoch Loss".format(modelType),
                   saveTo="{}/epoch_span_loss.png".format(graphdir))
        plot_learn(epoch_chunk_loss, "Loss", "Epochs", title="{} Epoch Loss".format(modelType),
                   saveTo="{}/epoch_chunk_loss.png".format(graphdir))
        plot_learn(epoch_f1, "F1", "Epochs", title="{} Epoch F1".format(modelType),
                   saveTo="{}/epoch_f1.png".format(graphdir))
        plot_learn(epoch_em, "Exact Match", "Epochs", title="{} Epoch EM".format(modelType),
                   saveTo="{}/epoch_em.png".format(graphdir))

        plot_metrics(epoch_vals, "Metrics (%)", "Epochs", title="{} Epoch Metrics".format(modelType),
                     saveTo="{}/epoch_all.png".format(graphdir))

        # plot_learn(iter_loss, "Loss", "Iterations", title="{} Iteration Loss".format(modelType), \
        #     saveTo="{}/iter_loss.png".format(graphdir))
        # plot_learn(iter_f1, "F1", "Iterations", title="{} Iteration F1".format(modelType), \
        #     saveTo="{}/iter_f1.png".format(graphdir))
        # plot_learn(iter_em, "Exact Match", "Iterations", title="{} Iteration EM".format(modelType), \
        #     saveTo="{}/iter_em.png".format(graphdir))
