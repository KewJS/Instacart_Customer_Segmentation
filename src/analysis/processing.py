import os, sys, fnmatch
import pandas as pd
import numpy as np
import argparse
from collections import OrderedDict
from datetime import datetime

from IPython.display import display, Markdown, HTML, clear_output, display_html

import matplotlib
import matplotlib.pyplot as plt
from termcolor import colored
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')

import hvplot
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
 
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from src.Config import Config
from src.analysis.feature_engineer import Feature_Engineer as fe
# from src.analysis.statistical_analysis import Statistical_Anlysis as sa

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print


class Analysis(Feature_Engineer):

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

        self.logger.info("    Renaming Columns ...")
        self.data["customer_data"] = self.data["customer_data"].rename(columns=self.COLUMN_RENAME["CUSTOMER"])

        self.logger.info("  Creating feature engineering ...")
        self.data["customer_data"]["peak_day"] = self.data['customer_data']['order_day_of_week'].apply(lambda x: self.peak_day_assignment(x))

        self.logger.info("    Creating number of products ordered across each days ...")
        self.data['prod_count_day_df'] = pd.DataFrame(self.data['customer_data'].groupby(['order_id', 'order_day_of_week']).count()['product_id'])
        self.data['prod_count_day_df'] = self.data['prod_count_day_df'].reset_index()
        self.data['prod_count_day_df'] = self.data['prod_count_day_df'].rename(columns={"product_id": "ordered_number"})
        self.data["prod_count_day_df"]["peak_day"] = self.data['prod_count_day_df']['order_day_of_week'].apply(lambda x: self.peak_day_assignment(x))

        if self.QDEBUG:
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PROD_COUNT_DAY"], self.suffix))
            self.data["prod_count_day_df"].to_csv(fname)

        self.logger.info("    Creating number of products ordered across each hours ...")
        self.data['prod_count_hour_df'] = pd.DataFrame(self.data['customer_data'].groupby(['order_id', 'order_hour_of_day']).count()['product_id'])
        self.data['prod_count_hour_df'] = self.data['prod_count_hour_df'].reset_index()
        self.data['prod_count_hour_df'] = self.data['prod_count_hour_df'].rename(columns={"product_id": "ordered_number"})

        if self.QDEBUG:
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PROD_COUNT_HOUR"], self.suffix))
            self.data["prod_count_hour_df"].to_csv(fname)

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

    
    # # holoviews
    def holoview_table(self, df, column, width, height):
        table = df.hvplot.table(columns=column, width=width, height=height)
        return table

    
    def histogram_plot_hv(self, df, title, var, bin_size=30, height=400, width=1000):
        frequencies, edges = np.histogram(df.dropna()[var], bins=bin_size)
        fig = hv.Histogram((edges, frequencies), label=title).opts(fontsize={'title': 16, 'labels': 12, 'xticks': 10, 'yticks': 10}, height=height, width=width, fill_color="#34495e")
        
        return fig


    # # bokeh
    def hist_hover(self, df, column, colors=["#34495e", "Tan"], bins=30, log_scale=False, show_plot=True):
        hist, edges = np.histogram(df.dropna()[column], bins = bins)
        hist_df = pd.DataFrame({column: hist,
                                "left": edges[:-1],
                                "right": edges[1:]})
        hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                                right in zip(hist_df["left"], hist_df["right"])]

        if log_scale == True:
            hist_df["log"] = np.log(hist_df[column])
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = 400, plot_width = 1000,
                        title = "Histogram of {}".format(column.capitalize()),
                        x_axis_label = column.capitalize(),
                        y_axis_label = "Log Count")    
            plot.quad(bottom = 0, top = "log",left = "left", 
                    right = "right", source = src, fill_color = "#34495e", 
                    line_color = "black", fill_alpha = 0.7,
                    hover_fill_alpha = 1.0, hover_fill_color = colors[1])
        else:
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = 400, plot_width = 1000,
                        title = "Histogram of {}".format(column.capitalize()),
                        x_axis_label = column.capitalize(),
                        y_axis_label = "Count")    
            plot.quad(bottom = 0, top = column,left = "left", 
                    right = "right", source = src, fill_color = "#34495e", 
                    line_color = "black", fill_alpha = 0.7,
                    hover_fill_alpha = 1.0, hover_fill_color = colors[1])
        
        hover = HoverTool(tooltips = [('Interval', '@interval'),
                                ('Count', str("@" + column))])
        plot.add_tools(hover)
        
        if show_plot == True:
            show(plot)
        else:
            return plot
        

    def histotabs(self, df, features, log_scale=False, show_plot=False):
        hists = []
        for f in features:
            h = self.hist_hover(df, f, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child=h, title=f.capitalize())
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)


    def filtered_histotabs(self, df, feature, filter_feature, log_scale=False, show_plot=False):
        hists = []
        for col in df[filter_feature].unique():
            sub_df = df[df[filter_feature] == col]
            histo = self.hist_hover(sub_df, feature, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child = histo, title=col)
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)

    
    # # matplotlib 
    def histogram_plot(self, df, title, var, xlabel, ylabel, bin_size=100, count=False):
        fig = plt.figure(figsize=(19,5))
        plt.suptitle(title, y=1, fontsize=18, weight='bold')
        
        if count:
            fig = plt.hist(df[var].value_counts(), bins=bin_size, color='#34495e')
            mean = plt.axvline(df[var].value_counts().mean(), color='orange', 
                    linestyle='--', label='Mean')
            median = plt.axvline(df[var].value_counts().median(), color='darkgreen', 
                        linestyle='--', label='Median')
        else:
            fig = plt.hist(df[var], bins=bin_size, color='#34495e')
            mean = plt.axvline(df[var].mean(), color='orange', 
                    linestyle='--', label='Mean')
            median = plt.axvline(df[var].median(), color='darkgreen', 
                        linestyle='--', label='Median')

        plt.title('Mean: {:.2f};    Median: {:.2f}'.format(df[var].value_counts().mean(), df[var].value_counts().median()))
        plt.xlabel(xlabel, weight='bold')
        plt.ylabel(ylabel, weight='bold')
        plt.legend(fontsize=13)
        sns.despine()
        plt.show()
        
        return fig

    
    def boxplot_plot_days(self, df, x_var, y_var, xlabel=None, ylabel=None, title=None):
        fig = plt.figure(figsize=(12, 8))
        fig = sns.boxplot(df[x_var], df[y_var], orient='h', color='#34495e')
        plt.yticks(range(7), [self.ANALYSIS_CONFIG["DAYS_ASSIGNED"][n] for n in range(7)])
        plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
        plt.show()
    
        return fig

    
    def violin_plot_hours(self, df, x_var, y_var, xlabel=None, ylabel=None, title=None):
        fig = plt.figure(figsize=(12, 8))
        fig = sns.violinplot(df[x_var], df[y_var], orient='h', color='#34495e')
        plt.yticks(range(24), [n for n in range(24)])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
        plt.show()
    
        return fig

    
    # # seaborn
    def count_plot(self, df, var, xlabel, title):
        fig, ax = plt.subplots()
        fig.set_size_inches(20,5)

        sns.countplot(color="#34495e", data=df, x=var, ax=ax)
        ax.set(xlabel=xlabel, title=title)
        
        return fig


def main():
    analysis = Analysis()
    analysis.get_data()


if __name__ == "__main__":
    main()