import os, sys, fnmatch
import pandas as pd
import numpy as np
import argparse
from collections import OrderedDict
from datetime import datetime

from IPython.display import display, Markdown, HTML, clear_output, display_html

import matplotlib
import hvplot
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
 

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from src.Config import Config

# from src.analysis.statistical_analysis import Statistical_Anlysis as sa
# from src.analysis.feature_engineer import Feature_Engineer as fe

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
    def vars(types=[], wc_vars=[], qreturn_dict=False):
        """ Return list of variable names
        
        Acquire the right features from dataframe to be input into model.  
        Featurs will be acquired based the value "predictive" in the VARS dictionary. 

        Parameters
        ----------
        types : str
            VARS name on type of features
        
        Returns
        -------
        Features with predictive == True in self.VARS
        """
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if not d.get('predictive'):
                    continue
                if len(wc_vars) != 0: 
                    matched_vars = fnmatch.filter(wc_vars, d['var'])
                    if qreturn_dict:
                        for v in matched_vars:
                            dd = d.copy()
                            dd['var'] = v 
                            if not dd in selected_vars:
                                selected_vars.append(dd)
                    else:
                        for v in matched_vars:
                            if not v in selected_vars:
                                selected_vars.append(v)
                else:
                    if qreturn_dict and not d in selected_vars:
                        selected_vars.append(d)
                    else:
                        if not d['var'] in selected_vars:
                            selected_vars.append(d['var'])
        return selected_vars

    
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
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["CUSTOMER_DATA"], self.suffix))
            self.data["customer_data"].to_csv(fname)

        # self.data["train_data"].to_csv(OUTPUT_PATH, "train_data_{}.csv".format(self.suffix))
        self.logger.info("done.")


    # # Data Processing
    def descriptive_data(self, df):
        descriptive_info = {"Number of User ID: ": df["user_id"].nunique(),
                            "No. of Variables: ": int(len(df.columns)),
                            "No. of Observations: ": int(df.shape[0]),
                            }
        descriptive_df = pd.DataFrame(descriptive_info.items(), columns=["Descriptions", "Values"]).set_index("Descriptions")
        descriptive_df.columns.names = ["Data Statistics"]
        return descriptive_df

    
    def data_type_analysis(self, df):
        categorical_df = pd.DataFrame(df.reset_index(inplace=False).dtypes.value_counts())
        categorical_df.reset_index(inplace=True)

        categorical_df = categorical_df.rename(columns={"index": "Types", 0:"Values"})
        categorical_df["Types"] = categorical_df["Types"].astype(str)
        categorical_df = categorical_df.set_index("Types")
        categorical_df.columns.names = ["Variables"]

        return categorical_df


    def grid_df_display(self, list_dfs, rows=1, cols=2, fill='cols'):
        html_table = "<table style = 'width: 100%; border: 0px'> {content} </table>"
        html_row = "<tr style = 'border:0px'> {content} </tr>"
        html_cell = "<td style='width: {width}%; vertical-align: top; border: 0px'> {{content}} </td>"
        html_cell = html_cell.format(width=8000)

        cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]
        cells += cols * [html_cell.format(content="")]

        if fill == 'rows':
            grid = [ html_row.format(content="".join(cells[i: i+cols])) for i in range(0,rows*cols, cols)]

        if fill == 'cols': 
            grid = [ html_row.format(content="".join(cells[i: rows*cols:rows])) for i in range(0,rows)]
            
        dfs = display(HTML(html_table.format(content="".join(grid))))
        return dfs

    
    def holoview_table(self, df, column, width, height):
        table = df.hvplot.table(columns=column, width=width, height=height)
        return table


def main():
    analysis = Analysis()
    analysis.get_data()


if __name__ == "__main__":
    main()