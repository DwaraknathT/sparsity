import errno
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.utils.args import get_args

args = get_args()

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s",
                              datefmt='%m/%d/%Y %I:%M:%S %p')


def get_console_handler():
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(FORMATTER)
  return console_handler


def get_file_handler(logfile_name):
  log_dir = 'runs/{}/'.format(args.output_dir)
  if not os.path.isdir(log_dir): os.makedirs(log_dir)
  if args.resume:
    filemode = 'a'
  else:
    filemode = 'w'

  file_handler = RotatingFileHandler('{}/{}.log'.format(log_dir, logfile_name), mode=filemode)
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


def setup_dirs():
  try:
    if args.use_colab:
      from google.colab import drive
      drive.mount('/content/gdrive')
      colab_str = '/content/gdrive/My Drive/sparsity/'
      OUTPUT_DIR = '{}/{}'.format(colab_str, args.output_dir)
      if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    else:
      os.makedirs('runs/{}'.format(args.output_dir))
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
