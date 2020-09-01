import os, sys, fnmatch
import time, inspect
from collections import OrderedDict
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

__current_module__ = "1.0.0"
__description__ = "Customer Segmentation of Retails Sales (Kaggle)"
__current_module__ = inspect.getfile(inspect.currentframe()).replace("__main__", "").replace("__.py", "")
__path__ = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)


class Parse(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(prog=os.path.split(__current_module__)[1],
                                              description="Description: " + __description__)
        subparsers = self.parser.add_subparsers(dest="task")
        subparsers.required = True

        analysis = subparsers.add_parser("analysis", help="perform data processing, transformation, feature engineering in the input dataset")
        analysis.add_argument("-c", "--customer", dest="customer", nargs="+", default=["*"],
                            help="selected product id(s) for which we choose to do the analysis, By default, all the product id(s) in the dataset.")
        # analysis.add_argument('-i', '--input', dest='input', default=Config.FILES["DATA_LOCAL"],
        #                     help="input directory containing raw data file; default is 'data_local'.")
        # analysis.add_argument('-s', '--suffix', dest='suffix', default='', 
        #                     help="suffix for in the input file name, default is '' means the file name will be named after the table names only.")