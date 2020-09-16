import pandas as pd
import numpy as np

from src.Config import Config

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print

class Feature_Engineer(Config):

    def peak_day_assignment(self, order_day):
        if order_day <= 1:
            return 1
        elif order_day > 1:
            return 0
        else:
            self.logger.info(order_day)