import fnmatch
import pandas as pd
import os, sys, inspect
from datetime import datetime, timedelta
from collections import OrderedDict

base_path, currentdir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):

    QDEBUG = True

    NAME = dict(
        full = "Customer Segmentation",
        short = "CS",
    )

    FILES = dict(
        DATA_LOCAL      = "data_local",
        
    )