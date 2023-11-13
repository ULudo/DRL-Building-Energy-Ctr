import os
import sys
import csv
import warnings
from logging import Logger, Formatter, FileHandler, \
    StreamHandler, Manager, basicConfig, INFO
import tensorflow as tf


class CsvFormatter:
    """Collects data records and writes them to
    csv files in a formatted structure.
    """
    
    def __init__(self, record_buffer=100) -> None:
        """
        Args:
            record_buffer (int, optional): Number of rows to buffer for one
                    csv file until writing them to csv. Defaults to 100.
        """
        self.threshold = record_buffer
        self.csv_records = {}
    
    
    def setup(self, log_dir):
        """Creates directory for csv files

        Args:
            log_dir (str): path to log folders
        """
        self.data_dir = os.path.join(log_dir, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    
    def _construct_csv_file_path(self, key_path):
        """Creates the csv file path in following formatted way:
            * If no record category: save in common csv file
            * If one record category: use the category as csv file name
                for these records
            * If more record categories: create them as folder structure
        
        Args:
            key_path (list): list of record categories

        Returns:
            str: csv file path
        """
        csv_file_dir = self.data_dir
        if len(key_path) == 1:
            # common CSV file for data with no subgroup
            csv_name = "common"
        else:
            # separate CSV file for each subgroup
            csv_name = key_path[-2]
            
            if len(key_path) > 2:
                # create further categorization as folder structure 
                csv_file_dir = os.path.join(csv_file_dir, *key_path[:-2])
                
                if not os.path.exists(csv_file_dir):
                    os.makedirs(csv_file_dir)
        
        return os.path.join(csv_file_dir, f"{csv_name}.csv")
        
    
    def _save_value_in_record_buffer(self, csv_file_dir, key, value, step):
        # create dictionary structure 
        if csv_file_dir not in self.csv_records:
            self.csv_records[csv_file_dir] = {}
        
        if step not in self.csv_records[csv_file_dir]:
            self.csv_records[csv_file_dir][step] = {}
        
        # save value
        self.csv_records[csv_file_dir][step][key] = value
    
    
    def _save_rows_to_csv(self, csv_file_dir, rows):
        if len(rows) > 0:
            csv_file_exists = os.path.isfile(csv_file_dir)
            fieldnames = rows[0].keys()
            with open(csv_file_dir, 'a', encoding='UTF8', newline='') as f:
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not csv_file_exists:
                    csv_writer.writeheader()
                
                csv_writer.writerows(rows)
    
    
    def record(self, key, value, step):
        """Save a data record in the csv buffer and flush
            flush buffer to csv is threshold is exceeded
        
        Args:
            key (str): record name
            value: record value
            step (int): record step
        """
        key_path = key.split('/')
        
        csv_file_dir = self._construct_csv_file_path(key_path)
        
        self._save_value_in_record_buffer(csv_file_dir, key_path[-1], value, step)
        
        # write data to csv file if threshold is reached
        if len(self.csv_records[csv_file_dir]) > self.threshold:
            
            # delete dict-items and save them as list
            csv_rows = []
            for k, v in list(self.csv_records[csv_file_dir].items())[:-1]:
                v["step"] = k
                csv_rows.append(v)
                del self.csv_records[csv_file_dir][k]
            
            self._save_rows_to_csv(csv_file_dir, csv_rows)
    
    
    def close(self):
        """Save all remaining records in the buffer to csv
        """
        for csv_file_dir in self.csv_records.keys():
            csv_rows = []
            for k, v in list(self.csv_records[csv_file_dir].items()):
                v["step"] = k
                csv_rows.append(v)
                del self.csv_records[csv_file_dir][k]

            self._save_rows_to_csv(csv_file_dir, csv_rows)


class RlLogger(Logger):

    def __init__(self, level):
        Logger.__init__(self, "rl", level) 


    def setup(self, log_dir:str, log_file, log_level, log_std_interval=1, log_tb_interval=1, flush_secs=120):
        self.log_dir = log_dir
        
        # set log level
        self.setLevel(log_level)

        # file format
        logFormatter = Formatter("[%(asctime)s] [%(levelname)-5.5s]  %(message)s")

        # log to file
        fileHandler = FileHandler(filename="{0}/{1}.log".format(log_dir, log_file), encoding='utf-8')
        fileHandler.setFormatter(logFormatter)
        self.addHandler(fileHandler)

        # log to console
        consoleHandler = StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        self.addHandler(consoleHandler)
        
        # create csv writer
        self.csv_writer = CsvFormatter()
        self.csv_writer.setup(log_dir)

        # tb writer
        self.tb_writer = tf.summary.create_file_writer(log_dir + "/tb", flush_millis=(flush_secs * 1000))
        
        self.log_std_interval = log_std_interval
        self.log_tb_interval = log_tb_interval
        self.debug("Logger initialized")


    def record(self, key, value, step):
        if isinstance(key, list):
            name = '/'.join(self)
        elif isinstance(key, str):
            name = key
        else:
            raise TypeError(key) 

        if step % self.log_tb_interval == 0:
            with self.tb_writer.as_default():
                tf.summary.scalar(name, data=value, step=step)

        if step % self.log_std_interval == 0:
            self.info(f"step = {step}: {name} = {value}")
        
        self.csv_writer.record(key, value, step)


    def text(self, key, text, step=0):
        with self.tb_writer.as_default():
            tf.summary.text(key, text, step)
        self.info(f"{key}: step = {step}: {text}")
    

    def close(self):
        self.csv_writer.close()
        
    
    def setup_eval_logger(self, name):
        self.eval_tb_writer = tf.summary.create_file_writer(self.log_dir + f"/tb/{name}")
    
    
    def eval_record(self, key, value, step):
        if isinstance(key, list):
            name = '/'.join(self)
        elif isinstance(key, str):
            name = key
        else:
            raise TypeError(key)
        
        with self.eval_tb_writer.as_default():
            tf.summary.scalar(name, data=value, step=step)
    
    
    def eval_text(self, key, text, step=0):
        with self.eval_tb_writer.as_default():
            tf.summary.text(key, text, step)
        self.info(f"{key}: step = {step}: {text}")
    

root = RlLogger(INFO)
Logger.root = root
Logger.manager = Manager(Logger.root)

#---------------------------------------------------------------------------
# Utility functions at module level.
# Need to be overwritten, to delegate everything to the rl logger.
#---------------------------------------------------------------------------

log_dir = ""

def getLogger(name=None):
    """
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
    """
    if name:
        return Logger.manager.getLogger(name)
    else:
        return root

def critical(msg, *args, **kwargs):
    """
    Log a message with severity 'CRITICAL' on the root logger. If the logger
    has no handlers, call basicConfig() to add a console handler with a
    pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.critical(msg, *args, **kwargs)

fatal = critical

def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.error(msg, *args, **kwargs)

def exception(msg, *args, exc_info=True, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger, with exception
    information. If the logger has no handlers, basicConfig() is called to add
    a console handler with a pre-defined format.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)

def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.warning(msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
    warnings.warn("The 'warn' function is deprecated, "
        "use 'warning' instead", DeprecationWarning, 2)
    warning(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.info(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.debug(msg, *args, **kwargs)

def log(level, msg, *args, **kwargs):
    """
    Log 'msg % args' with the integer severity 'level' on the root logger. If
    the logger has no handlers, call basicConfig() to add a console handler
    with a pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.log(level, msg, *args, **kwargs)

def record(key, value, step):
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.record(key, value, step)
    
def close():
    root.close()


def setup_eval_logger(name):
    root.setup_eval_logger(name)


def eval_record(key, value, step):
    if len(root.handlers) == 0:
        basicConfig()
    root.eval_record(key, value, step)


def eval_text(key, text, step=0):
    root.eval_text(key, text, step)


def text(key, text, step=0):
    root.text(key, text, step)
