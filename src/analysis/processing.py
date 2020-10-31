import os, sys, fnmatch, random, json, logging, argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
from datetime import datetime
from dateutil import parser
from IPython.display import display, Markdown, HTML, clear_output, display_html
from operator import itemgetter
from jinja2 import Template
from urllib.parse import urlparse
from urllib.request import urlopen

import scipy.stats as stats
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
from scipy.stats import chi2, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox, shapiro, gaussian_kde

import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Enum

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
pn.extension()
from io import StringIO
from bokeh.io import show, curdoc
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models.filters import CustomJSFilter
from bokeh.layouts import column, row, WidgetBox, gridplot
from bokeh.palettes import Category10_10, Category20_16, Category20_20, Category20
from bokeh.models import Column, CDSView, CustomJS, CategoricalColorMapper, ColumnDataSource, HoverTool, Panel, MultiSelect
from bokeh.models.widgets import CheckboxGroup, CheckboxButtonGroup, Slider, RangeSlider, Tabs, TableColumn, DataTable

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from src.Config import Config
from src.analysis.feature_engineer import Feature_Engineer
from src.analysis.statistical_analysis import Statistic_Analysis

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print


class JSON_Datetime_Encoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        else:
            return json.JSONEncoder.default(self, obj)        


class Analysis(Feature_Engineer, Statistic_Analysis):

    data = {}

    json = JSON_Datetime_Encoder()

    def __init__(self, customer_id=["*"], suffix="", logger=Logger()):
        self.customer_id = customer_id
        self.suffix = suffix
        self.logger = logger 

    
    @staticmethod
    def style(p):
        # # Title
        p.title.align = "center"
        p.title.text_font_size = "16pt"
        p.title.text_font = "serif"

        # # Axis Titles
        p.xaxis.axis_label_text_font_size = "12pt"
        p.xaxis.axis_label_text_font_style = "bold"
        p.yaxis.axis_label_text_font_size = "12t"
        p.yaxis.axis_label_text_font_style = "bold"

        # # Tick Labels
        p.xaxis.major_label_text_font_size = "10pt"
        p.yaxis.major_label_text_font_size = "10pt"

        return p

    
    @staticmethod
    def get_template(path):
        if bool(urlparse(path).netloc):
            from urllib.request import urlopen
            return urlopen(path).read().decode('utf8')
        return open(path).read()

    
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

    
    def dump_json(self, obj):
        return json.dumps(obj, cls=JSON_Datetime_Encoder)


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
            self.logger.error("Please select only 'single' or 'multiple' ...")
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

        self.logger.info("  Data Processing ...")
        self.logger.info("   renaming columns ...")
        self.data["customer_data"] = self.data["customer_data"].rename(columns=self.COLUMN_RENAME["CUSTOMER"])
        self.data["customer_data"]["user_id"] = self.data["customer_data"]["user_id"].astype(str)

        # # Randomly select 20 customer from the ID for analysis, as bokeh palettes only allow maximum 20 subset
        self.available_group = list(self.data["customer_data"]['user_id'].unique())
        self.available_group = random.sample(self.available_group, k=20)

        # # Data correlation
        self.logger.info("  Generating data correlation dataframe for feature dependency ...")
        self.data["data_correlation_df"] = self.data["customer_data"][self.vars(["Customer"], self.data["customer_data"].columns)]

        if self.QDEBUG:
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["DATA_CORRELATION"], self.suffix))
            self.data["data_correlation_df"].to_csv(fname)

            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["CUSTOMER_DATA"], self.suffix))
            self.data["customer_data"].to_csv(fname)

        self.logger.info("done.")

    
    def feature_engineering(self):
        self.logger.info("Creating new feature ...")
        self.logger.info("  Creating number of products ordered across each days ...")
        self.data['day_peak_df'] = self.data['customer_data'].groupby(['order_id', 'order_day_of_week'])['order_number'].count().reset_index().rename(columns={'order_number': 'count'})
        self.data["day_peak_df"]["peak_day"] = np.where(self.data['day_peak_df']['order_day_of_week'] <= 1, 1, 0)

        self.logger.info("  Creating number of products ordered across each hours ...")
        self.data['time_peak_df'] = self.data['customer_data'].groupby(['order_id', 'order_hour_of_day'])['order_number'].count().reset_index().rename(columns={'order_number': 'count'})
        self.data["time_peak_df"]["peak_time"] = np.where((self.data["time_peak_df"]['order_hour_of_day'] >= self.ANALYSIS_CONFIG["PEAK_DAY_FROM"]) & 
                                                          (self.data["time_peak_df"]['order_hour_of_day'] <= self.ANALYSIS_CONFIG["PEAK_DAY_TO"]), 1, 0)

        # # RFM features
        self.logger.info("  Creating number of products ordered in each order ...")
        num_products = self.data["customer_data"].groupby(['order_id'])['product_id'].count().reset_index().rename(columns={'product_id':'num_products'})
        self.data["customer_data"] = pd.merge(self.data["customer_data"], num_products, on='order_id', how='left')

        self.logger.info("  Creating peak day categorical feature ...")
        self.data["customer_data"]['peak_day'] = np.where(self.data["customer_data"]['order_day_of_week'] <= 1, 1, 0)
        self.logger.info("  Creating peak time categorical feature (from 10 - 16) ...")
        self.data["customer_data"]['peak_time'] = np.where((self.data["customer_data"]['order_hour_of_day'] >= self.ANALYSIS_CONFIG["PEAK_DAY_FROM"]) & 
                                                           (self.data["customer_data"]['order_hour_of_day'] <= self.ANALYSIS_CONFIG["PEAK_DAY_TO"]), 1, 0)
        
        self.logger.info("  Creating number of orders per customer, peak day rate, median hour, peak time rate, mean lag days since last order, mean number of products ...")
        num_orders          = self.data["customer_data"].groupby(['user_id'])['order_number'].max()
        peakday_rate        = round(self.data["customer_data"].groupby(["user_id"])['peak_day'].mean(), 2)
        med_hour            = round(self.data["customer_data"].groupby('user_id')['order_hour_of_day'].median(), 0)
        peaktime_rate       = round(self.data["customer_data"].groupby(['user_id'])['peak_time'].mean(), 2)
        mean_lag_days       = round(self.data["customer_data"].groupby(['user_id'])['days_since_last_order'].mean(), 0)
        mean_num_products   = round(self.data["customer_data"].groupby('user_id')['num_products'].mean(), 0)

        self.data['features'] = pd.concat([num_orders, peakday_rate, med_hour, peaktime_rate, mean_lag_days, mean_num_products], axis=1)
        self.data['features'].columns = self.ANALYSIS_CONFIG["FEATURES_COL"]
        self.data['features'] = self.data['features'].reset_index()
        self.data["customer_data"] = pd.merge(self.data["customer_data"], self.data['features'], on='user_id', how='left')

        random.seed(30)
        self.data["sample_df"] = self.data["customer_data"].sample(1000)

        self.logger.info("  Creating number of purchase across each products ...")
        self.data["product_count_df"] = self.data["sample_df"].groupby(['product_id', 'product_name'])['num_orders'].count().reset_index().rename(columns={'num_orders': 'count'}).sort_values(by="count", ascending=False)

        self.logger.info("  Creating number of purchase across each products groups ...")
        self.data["prod_gp_count_df"] = self.data["sample_df"].groupby(['product_group_id', 'product_group'])['num_orders'].count().reset_index().rename(columns={'num_orders': 'count'}).sort_values(by="count", ascending=False)

        self.logger.info("  Creating customer data overview for data summary display ...")
        self.data["df_table"], self.data["overall_summary"] = self.get_overview_table(self.data["sample_df"], 'user_id', group="user")
        self.logger.info("  Creating dictionary info on sum of orders, count of orders and mean of peak hour ...")
        self.data["sum_order_summary"], self.data["count_order_summary"], self.data["mean_hour_order_summary"] = self.get_overview_table(self.data["sample_df"], "product_group", group="product")
        self.logger.info("  Extracting data overview template file ...")
        self.overview_template = Template(self.get_template(self.FILES["TEMPLATE_PATH"]))

        if self.QDEBUG:
            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PROD_COUNT_DAY"], self.suffix))
            self.data["day_peak_df"].to_csv(fname)

            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PROD_COUNT_HOUR"], self.suffix))
            self.data["time_peak_df"].to_csv(fname)

            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["FEATURES"], self.suffix))
            self.data["features"].to_csv(fname)

            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PRODUCT_ORDERED"], self.suffix))
            self.data["product_count_df"].to_csv(fname)

            fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["PRODUCT_GROUP_ORDERED"], self.suffix))
            self.data["prod_gp_count_df"].to_csv(fname)

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

    
    def get_overview_table(self, df, group_var, group="user"):
        if group == "user":
            user_num_of_order                  = df.groupby([group_var])['order_number'].sum()
            user_peak_order_day_mean           = df.groupby([group_var])['order_day_of_week'].mean()
            user_peak_order_hour_mean          = df.groupby([group_var])['order_hour_of_day'].mean()
            user_num_of_products               = df.groupby([group_var])['num_products'].sum()
            user_mean_of_days_from_last_orders = df.groupby([group_var])['days_since_last_order'].mean()

            df_table = (pd.DataFrame(dict(
                num_of_order=user_num_of_order, peak_order_day_mean=user_peak_order_day_mean, peak_order_time_mean=user_peak_order_hour_mean,
                num_of_products=user_num_of_products, days_since_last_orders_mean=user_mean_of_days_from_last_orders)).sort_values(by=['num_of_order'], ascending=False).reset_index())
            
            overall_summary                     = {}
            overall_summary['Number of Users']  = df[group_var].count()
            overall_summary['Number of Orders'] = df['num_orders'].sum()
            return df_table, overall_summary
        
        elif group == "product":
            sum_of_order_summary    = df.groupby([group_var])['order_number'].sum().to_dict()
            count_order_summary     = df.groupby([group_var])['order_number'].count().to_dict()
            mean_hour_order_summary = df.groupby([group_var])['order_hour_of_day'].mean().round(2).to_dict()
            
            return sum_of_order_summary, count_order_summary, mean_hour_order_summary
        
        else:
            self.logger.error("Invalid group, please key in only 'user' or 'product'.")

    
    # # Data Visualization
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


    def get_template(self, path):
        from urllib.parse import urlparse
        if bool(urlparse(path).netloc):
            from urllib.request import urlopen
            return urlopen(path).read().decode('utf8')
        return open(path).read()


    # # matplotlib 
    def histogram_plot(self, df, title, var, xlabel, ylabel, bin_size=100, count=False):
        fig, ax = plt.subplots(figsize=(19,5))
        plt.suptitle(title, y=1, fontsize=18, weight='bold')
        
        if count:
            plt.hist(df[var].value_counts(), bins=bin_size, color='#34495e')
            mean = plt.axvline(df[var].value_counts().mean(), color='orange', linestyle='--', label='Mean')
            median = plt.axvline(df[var].value_counts().median(), color='darkgreen', linestyle='--', label='Median')
            plt.title('Mean: {:.2f};    Median: {:.2f}'.format(df[var].value_counts().mean(), df[var].value_counts().median()))
        else:
            plt.hist(df[var], bins=bin_size, color='#34495e')
            mean = plt.axvline(df[var].mean(), color='orange', linestyle='--', label='Mean')
            median = plt.axvline(df[var].median(), color='darkgreen', linestyle='--', label='Median')
            plt.title('Mean: {:.2f};    Median: {:.2f}'.format(df[var].mean(), df[var].median()))
        
        plt.xlabel(xlabel, weight='bold')
        plt.ylabel(ylabel, weight='bold')
        plt.legend(fontsize=13)
        sns.despine()
        
        return fig

    
    def horizontal_bar_plot(self, df, x_var, y_var, xlabel, ylabel, title, xticks=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=x_var, y=y_var, data=df, color='#34495e')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if xticks == "product":
            plt.xticks(ticks=[0, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4], labels=['0', '10k', '20k', '30k', '40k', '50k', '60k'])
        elif xticks == "department":
            plt.xticks(ticks=[0, 2e5, 4e5, 6e5, 8e5, 1e6, 1.2e6], labels=['0', '200k', '400k', '600k', '800k', '1m', '1.2m'])
        else:
            pass
        
        plt.tight_layout()
        sns.despine()
        
        return fig

    
    def boxplot_plot(self, df, x_var, y_var, xlabel=None, ylabel=None, title=None, day=True):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(df[x_var], df[y_var], orient='h', color='#34495e')
        if day:
            plt.yticks(range(7), [self.ANALYSIS_CONFIG["DAYS_ASSIGNED"][n] for n in range(7)])
        else:
            plt.yticks(range(24), [n for n in range(24)])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
    
        return fig

    
    def violin_plot_hours(self, df, x_var, y_var, xlabel=None, ylabel=None, title=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(df[x_var], df[y_var], orient='h', color='#34495e')
        plt.xticks(range(24), [n for n in range(24)])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()
    
        return fig

    
    def horizontal_stack_plot(self, df, var1, var2):
        count_data = pd.crosstab(df[var1], df[var2]).T

        category_names = count_data.columns.tolist()
        labels = count_data.index.tolist()
        data = count_data.values
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[0]))

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(labels, widths, left=starts, height=0.8, label=colname, color=color)
            xcenters = starts + widths / 2
        #     r, g, b, _ = color
        #     text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #     for y, (x, c) in enumerate(zip(xcenters, widths)):
        #         ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color)

        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left')
        
        return fig

    
    # # seaborn
    def count_plot(self, df, var, xlabel, title):
        fig, ax = plt.subplots()
        fig.set_size_inches(20,5)

        sns.countplot(color="#34495e", data=df, x=var, ax=ax)
        ax.set(xlabel=xlabel, title=title)
        
        return fig


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

    
    def table_tab(self, df, impt_var, subset_var):
        df_stats = df.groupby(subset_var)[impt_var].describe()
        df_stats = df_stats.reset_index().rename(columns={'50%': 'median'})

        df_stats['mean'] = df_stats['mean'].round(2)
        df_src = ColumnDataSource(df_stats)

        table_columns = [TableColumn(field='user_id', title='User ID'),
                        TableColumn(field='count', title='Count'),
                        TableColumn(field='min', title='Min of {}'.format(impt_var)),
                        TableColumn(field='mean', title='Mean of {}'.format(impt_var)),
                        TableColumn(field='median', title='Median of {}'.format(impt_var)),
                        TableColumn(field='max', title='Max of {}'.format(impt_var))]

        info_table = DataTable(source=df_src, columns=table_columns, width=1000)

        tab = Panel(child=info_table, title='Summary Table')

        return tab

    
    def csv_download(self, df, group_var, range_var):
        subset_group = pn.widgets.MultiChoice(
            name="Subset Variable", options=list(df[group_var].unique()), margin=(0, 20, 0, 0)
        )

        num_range = pn.widgets.RangeSlider(
            name="Numeric Range Var", start=df[range_var].min(), end=df[range_var].max()
        )

        @pn.depends(subset_group, num_range)
        def filtered_group(gv, rv):
            temp_df = df
            if subset_group.value:
                temp_df = df[df[group_var].isin(gv)]
            return temp_df[(temp_df[range_var]>=rv[0]) & (temp_df[range_var]<=rv[1])]

        @pn.depends(subset_group, num_range)
        def filtered_file(gv, rv):
            temp_df = filtered_group(gv, rv)
            sio = StringIO()
            temp_df.to_csv(sio)
            sio.seek(0)
            return sio

        fd = pn.widgets.FileDownload(
            callback = filtered_file, filename="subset_df.csv"
        )

        tab = pn.Column(pn.Row(subset_group, num_range), fd, pn.panel(filtered_group, width=2600), width=600).servable()
        
        return tab

    
    def histo_tab(self, df, impt_var, subset_var, subset_list, range_start, range_end, bin_width):
        def make_dataset(subset_list, range_start, range_end, bin_width):
            by_subset = pd.DataFrame(columns=['proportion', 'left', 'right', 'f_proportion', 'f_interval','group', 'color'])
            range_extent = range_end - range_start

            for i, subset in enumerate(subset_list):
                subset_df = df[df[subset_var]==subset]
                arr_hist, edges = np.histogram(subset_df[impt_var], bins=bin_width, range=[range_start, range_end])

                arr_df = pd.DataFrame({'proportion': arr_hist / np.sum(arr_hist), 'left': edges[:-1], 'right': edges[1:] })
                arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]
                arr_df['f_interval'] = ['%0.2f to %0.2f' % (left, right) for left, right in zip(arr_df['left'], arr_df['right'])]
                arr_df['group'] = subset
                arr_df['color'] = Category20_20[i]

                by_subset = by_subset.append(arr_df)

            by_subset = by_subset.sort_values(['group', 'left'])
            return ColumnDataSource(by_subset)

        def make_plot(src):
            p = figure(plot_width=700, plot_height=700, title='Histogram of {}'.format(impt_var), x_axis_label=impt_var, y_axis_label="Density")

            p.quad(source=src, bottom=0, top='proportion', left='left', right='right',
                    color='color', fill_alpha=0.7, hover_fill_color='color', legend='group',
                    hover_fill_alpha=1.0, line_color='black')

            hover = HoverTool(tooltips=[('Group', '@group'), 
                                        ('Interval', '@f_interval'),
                                        ('Proportion', '@f_proportion')],
                            mode='vline')

            p.add_tools(hover)
            p.legend.click_policy = 'hide'
            p = self.style(p)
            return p

        def update(attr, old, new):
            group_to_plot = [group_selection.labels[i] for i in group_selection.active]

            new_src = make_dataset(group_to_plot, range_start=range_select.value[0], range_end=range_select.value[1], bin_width=binwidth_select.value)

            src.data.update(new_src.data)

        group_selection = CheckboxGroup(labels=subset_list, active=[0, 1])
        initial_group = [group_selection.labels[i] for i in group_selection.active]
        group_selection.on_change('active', update)

        binwidth_select = Slider(start=1, end=30, step=1, value=5, title='Bin Size')
        binwidth_select.on_change('value', update)

        range_select = RangeSlider(start=range_start, end=range_end, value=(range_start+5, range_start+10),
                                step=5, title = 'Range Slider')
        range_select.on_change('value', update)
        
        src = make_dataset(initial_group, range_start=range_select.value[0], range_end=range_select.value[1], bin_width=binwidth_select.value)
        p = make_plot(src)

        controls = WidgetBox(group_selection, binwidth_select, range_select)
        layout = row(controls, p)
        tab = Panel(child=layout, title = 'Histogram Plot')

        return tab

    def density_tab(self, df, impt_var, subset_var, subset_list, range_start, range_end, bandwidth):
        def make_dataset(subset_list, range_start, range_end, bandwidth):

            xs = []
            ys = []
            colors = []
            labels = []

            for i, subset in enumerate(subset_list):
                subset_df = df[df[subset_var]==subset]
                subset_df = subset_df[subset_df[impt_var].between(range_start, range_end)]
                kde = gaussian_kde(subset_df[impt_var], bw_method=bandwidth)
                
                x = np.linspace(range_start, range_end, 50)
                y = kde.pdf(x)
                xs.append(list(x))
                ys.append(list(y))

                colors.append(Category20_20[i])
                labels.append(subset)

            new_src = ColumnDataSource(data={'x': xs, 'y': ys, 'color': colors, 'label': labels})
            return new_src

        def make_plot(src):
            p = figure(plot_width=700, plot_height=700, title='Density Plot of {}'.format(impt_var), x_axis_label=impt_var, y_axis_label='Density')

            p.multi_line('x', 'y', color='color', legend='label', line_width=3, source=src)

            hover = HoverTool(tooltips=[('Group', '@label'), 
                                        ('Variable', '$x'),
                                        ('Density', '$y')],
                            line_policy = 'next')

            p.add_tools(hover)
            p = self.style(p)

            return p
        
        def update(attr, old, new):
            group_to_plot = [group_selection.labels[i] for i in group_selection.active]
            
            if bandwidth_choose.active == []:
                bandwidth = None
            else:
                bandwidth = bandwidth_select.value
            
            new_src = make_dataset(group_to_plot, range_start=range_select.value[0], range_end=range_select.value[1], bandwidth=bandwidth)
            
            src.data.update(new_src.data)

        group_selection = CheckboxGroup(labels=subset_list, active=[0, 1])
        group_selection.on_change('active', update)
        
        range_select = RangeSlider(start=range_start, end=range_end, value=(range_start+5, range_start+10),
                                step=5, title = 'Range Slider')
        range_select.on_change('value', update)
        
        initial_group = [group_selection.labels[i] for i in group_selection.active]
        
        bandwidth_select = Slider(start= 0.1, end=5, step= 0.1, value=0.5, title='Bandwidth for Density Plot')
        bandwidth_select.on_change('value', update)
        
        bandwidth_choose = CheckboxButtonGroup(labels=['Choose Bandwidth (Else Auto)'], active = [])
        bandwidth_choose.on_change('active', update)

        src = make_dataset(initial_group, 
                            range_start=range_select.value[0],
                            range_end=range_select.value[1],
                            bandwidth=bandwidth_select.value) 
        
        p = make_plot(src)
        p = self.style(p)
        controls = WidgetBox(group_selection, range_select, bandwidth_select, bandwidth_choose)
        layout = row(controls, p)
        tab = Panel(child=layout, title = 'Density Plot')
        # tabs = Tabs(tabs=[tab])
        # curdoc().add_root(tabs)
                    
        return tab