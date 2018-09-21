"""
Module to handle universal/general constants used across files.
"""

################################################################################
# Constants #
################################################################################

# GENERAL CONSTANTS:

_UNK_TOKEN = '<<unk>>'
_NO_ANSWER = '<<no_answer>>'

# LOG FILES ##

_CONFIG_FILE = "config.json"
_LOG_FILE = "exp.log"
_SAVED_WEIGHTS_FILE = "params.saved"
_PREDICTION_FILE = "predictions.json"
_SAVED_ERROR_LOG_FILE = "error_log.json"

_TRAIN_LOSS_ITER_LOG = "metrics/train_loss_iter.txt"
_TRAIN_F1_ITER_LOG = "metrics/train_f1_iter.txt"
_TRAIN_EM_ITER_LOG = "metrics/train_em_iter.txt"

# _DEV_LOSS_ITER_LOG = "metrics/dev_loss_iter.txt"
_DEV_F1_ITER_LOG = "metrics/dev_f1_iter.txt"
_DEV_EM_ITER_LOG = "metrics/dev_em_iter.txt"

_TRAIN_LOSS_EPOCH_LOG = "metrics/train_loss_epoch.txt"
_TRAIN_F1_EPOCH_LOG = "metrics/train_f1_epoch.txt"
_TRAIN_EM_EPOCH_LOG = "metrics/train_em_epoch.txt"
_TRAIN_EPOCH_TIME_LOG = "metrics/train_time_epoch.txt"

# _DEV_LOSS_EPOCH_LOG = "metrics/dev_loss_epoch.txt"
_DEV_F1_EPOCH_LOG = "metrics/dev_f1_epoch.txt"
_DEV_EM_EPOCH_LOG = "metrics/dev_em_epoch.txt"

_TEST_EVAL_LOG = "metrics/test_scores.txt"
_ERROR_EXS = "error_exs.txt"
