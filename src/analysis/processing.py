import os, sys, fnmatch
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import datetime

import matplotlib

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from src.analysis.statistical_analysis import Statistical_Anlysis as sa
from src.analysis.feature_engineer import Feature_Engineer as fe

FILE_PATH = r"D:\03-Training\Digital\03-SelfLearning\29-Customer_Segmentation\data_local"
QDEBUG = True
OUTPUT_PATH = r"D:\03-Training\Digital\03-SelfLearning\29-Customer_Segmentation\data_local\processed_data"

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print


class Analysis(object):

    data = {}

    def __init__(self, customer_id=["*"], suffix="", logger=Logger()):
        self.customer_if = customer_id
        self.suffix = suffix
        self.logger = logger

    
    def read_file(self, fname, source_type="single"):
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
                fname = "{}.csv".format(os.path.join(FILE_PATH, fname))
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
            for root, dirs, files in os.walk(os.path.join(FILE_PATH)):
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

        self.logger.info("  Merging dataframe ...")
        self.logger.info("    on Orders-Prior & Orders:")
        self.data["customer_data"] = pd.merge(self.data["order_prior_df"], self.data["order_df"], on=["order_id"], how="left")
        self.logger.info("    on Customer Orders & Products:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["products_df"], on=["product_id"], how="left")
        self.logger.info("    on Customer Orders & Aisle:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["aisles_df"], on=["aisle_id"], how="left")
        self.logger.info("    on Customer Orders & Departments:")
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data["departments_df"], on=["department_id"], how="left")

        if QDEBUG:
            fname = os.path.join(OUTPUT_PATH, "customer_data{}.csv".format(self.suffix))

        self.logger.info("done.")


    def get_order(self):
        self.logger.info("  Loading order dataframe ...")
        self.data["order_df"] = self.read_file(fname="orders", source_type="single")

        self.logger.info("   Data information ...")
        self.logger.info(self.data["order_df"].info())


    def get_order_prior(self):
        self.logger.info("  Loading order prior dataframe ...")
        self.data["order_prior_df"] = self.read_file(fname="order_products__prior", source_type="single")

        self.logger.info("   Data information ...")
        self.logger.info(self.data["order_prior_df"].info())


    def get_products(self):
        self.logger.info("  Loading products dataframe ...")
        self.data["products_df"] = self.read_file(fname="products", source_type="single")

        self.logger.info("   Data information ...")
        self.logger.info(self.data["products_df"].info())

    
    def get_departments(self):
        self.logger.info("  Loading departments dataframe ...")
        self.data["departments_df"] = self.read_file(fname="departments", source_type="single")

        self.logger.info("   Data information ...")
        self.logger.info(self.data["departments_df"].info())


    def get_aisles(self):
        self.logger.info("  Loading aisles dataframe ...")
        self.data["aisles_df"] = self.read_file(fname="aisles", source_type="single")

        self.logger.info("   Data information ...")
        self.logger.info(self.data["aisles_df"].info())

