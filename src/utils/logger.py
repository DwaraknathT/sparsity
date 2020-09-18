import errno
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s",
                              datefmt='%m/%d/%Y %I:%M:%S %p')


def get_console_handler():
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(FORMATTER)
  return console_handler


def get_file_handler(logfile_name):
  try:
    file_handler = RotatingFileHandler('logs/{}.log'.format(logfile_name, mode='w'))
  except:
    raise OSError('Logs directory not created')
  file_handler.setFormatter(FORMATTER)
  return file_handler


def get_logger(logger_name):
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
  logger.addHandler(get_console_handler())
  logger.addHandler(get_file_handler(logger_name))
  # with this pattern, it's rarely necessary to propagate the error up to parent
  logger.propagate = False
  return logger


def setup_dirs(hparams):
  log_dir = 'logs'
  runs_dir = 'runs'
  try:
    if hparams.use_colab:
      from google.colab import drive
      drive.mount('/content/gdrive')
      colab_str = '/content/gdrive/My Drive/sparsity/'
      OUTPUT_DIR = '/{}'.format(hparams.output_dir)
      if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    else:
      os.makedirs('logs/{}'.format(hparams.output))
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
