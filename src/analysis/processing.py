import os, sys, fnmatch
import pandas as pd
import numpy as np
import argparse
from collections import OrderedDict
from datetime import datetime

import matplotlib

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from src.Config import Config

# from src.analysis.statistical_analysis import Statistical_Anlysis as sa
# from src.analysis.feature_engineer import Feature_Engineer as fe

# QDEBUG = True
# FILES_TYPE = "single"
# FILE_PATH = r"D:\03-Training\Digital\03-SelfLearning\29-Customer_Segmentation\data_local"
# OUTPUT_PATH = r"D:\03-Training\Digital\03-SelfLearning\29-Customer_Segmentation\data_local\processed_data"

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print


class Analysis(Config):

    data = {}

    def __init__(self, customer_id=["*"], suffix="", logger=Logger()):
        self.customer_id = customer_id
        self.suffix = suffix
        self.logger = logger 

    
    @staticmethod
    def vars(var_type=None, wc_vars=[], qpredictive=False):
        """Return a list of variable names

        Parameters
        ----------
        var_type : [type]
            Group of features, by default None
        wc_vars : list, optional
            [description], by default []
        qpredictive : bool, optional
            [description], by default False
        """
        if var_type == None:
            var_type = [V for V in Config.VARS]
        
        selected_vars = []
        for t in var_type:
            for d in Config.VARS[t]:
                if qpredictive and d.get("predictive", False):
                    pass
                elif len(wc_vars) != 0:
                    selected_vars.extend(fnmatch.filter(wc_vars, d['var']))
                else:
                    selected_vars.append(d['var'])
        return list(set(selected_vars))

    
    def read_file(self, fname, source_type=Config.ANALYSIS_CONFIG["FILE_TYPE"]):
        """Read in files, focusing on csv files.

        Parameters
        ----------
        fname : [type]
            [description]
        source_type : str, optional
            [description], by default "single"

        Returns
        -------
        [type]
            [description]
        """
        if source_type == "single":
            try:
                fname = "{}.csv".format(os.path.join(self.FILES["DATA_LOCAL"], fname))
                data = pd.read_csv(fname, error_bad_lines=False)
                if data.size == 0:
                    self.logger.warning("no data found in file {}".format(fname))
                    if self.logger.warning == print:
                        exit()
            except FileNotFoundError:
                self.logger.critical("file {} is not found ...".format(fname))
                if self.logger.critical == print:
                    exit()
        elif source_type == "multiple":
            csv_ext = [".csv"]
            data = pd.DataFrame()
            for root, dirs, files in os.walk(os.path.join(self.FILES["DATA_LOCAL"])):
                for filename in files:
                    if filename.endswith(tuple(csv_ext)):
                        df_temp = pd.read_csv(os.path.join(root, filename))
                    data = pd.concat([data, df_temp], axis=0, sort=True)
        else:
            self.logger.info("Please select only 'single' or 'multiple' ...")
        return data


    def get_data(self):
        self.logger.info("Reading in data ...")

        self.logger.info("  Loading order dataframe ...")
        self.data["order_df"] = self.read_file(fname="orders", source_type="single")

        self.logger.info("  Loading order prior dataframe ...")
        self.data["order_prior_df"] = self.read_file(fname="order_products__prior", source_type="single")

        self.logger.info("  Loading products dataframe ...")
        self.data["products_df"] = self.read_file(fname="products", source_type="single")

        self.logger.info("  Loading departments dataframe ...")
        self.data["departments_df"] = self.read_file(fname="departments", source_type="single")

        self.logger.info("  Loading aisles dataframe ...")
        self.data["aisles_df"] = self.read_file(fname="aisles", source_type="single")

        self.logger.info("   Merging dataframe ...")
        self.logger.info("     on Orders-Prior & Orders:")
        self.data["customer_data"] = pd.merge(self.data["order_prior_df"], self.data["order_df"], on=["order_id"], how="left")
        self.logger.info("     on Customer Orders & Products:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["products_df"], on=["product_id"], how="left")
        self.logger.info("     on Customer Orders & Aisle:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["aisles_df"], on=["aisle_id"], how="left")
        self.logger.info("     on Customer Orders & Departments:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["departments_df"], on=["department_id"], how="left")

        self.logger.info("  Renaming Columns ...")
        self.data["customer_data"] = self.data["customer_data"].rename(columns=self.COLUMN_RENAME["CUSTOMER"])

        self.logger.info("  Generating data correlation dataframe for feature dependency ...")
        self.data["data_correlation_df"] = self.data["customer_data"][self.vars(["Customer"], self.data["customer_data"].columns)]

        if self.QDEBUG:
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "customer_data{}.csv".format(self.suffix))

        # self.data["train_data"].to_csv(OUTPUT_PATH, "train_data_{}.csv".format(self.suffix))
        self.logger.info("done.")


