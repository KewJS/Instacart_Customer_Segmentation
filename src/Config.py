import fnmatch
import pandas as pd
import os, sys, inspect
from datetime import datetime, timedelta
from collections import OrderedDict

base_path, currentdir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):

    QDEBUG = False

    NAME = dict(
        full = "Customer Segmentation",
        short = "CS",
    )

    FILES = dict(
        DATA_LOCAL             = "data_local",
        OUTPUT_PATH            = r"D:\03-Training\Digital\03-SelfLearning\29-Customer_Segmentation\data_local\processed_data",
        TEMPLATE_PATH          = r"D:\03-Training\Digital\03-SelfLearning\29b_Customer_Segmentation\customer_segmentation\src\assets\templates\customer_overview.jinja", 
        CUSTOMER_DATA          = "customer_data",
        PROD_COUNT_DAY         = "order_count_day_data",
        PROD_COUNT_HOUR        = "order_count_hour_data",
        DATA_CORRELATION       = "data_correlation_data",
        FEATURES               = "features_data",
        PRODUCT_ORDERED        = "product_order_data",
        PRODUCT_GROUP_ORDERED  = "product_group_order_data",
        TRAIN_DATA             = "train_data",
    )

    ANALYSIS_CONFIG = dict(
        FILE_TYPE           = "single",
        DAYS_ASSIGNED       = {0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'},
        LEVENE_DISTRIBUTION = "Mean",
        TEST_ALPHA          = 0.05,
        SAMPLE_SIZE         = 1000,
        VIS_CONTROL_GROUP   = "user_id",
        PEAK_DAY_FROM       = 10,
        PEAK_DAY_TO         = 16,
        FEATURES_COL        = ['num_orders', 'peakday_rate', 'median_hour', 'peaktime_rate', 'mean_lag_days', 'mean_num_products'],

    )

    MODELLING_CONFIG = dict(

    )

    COLUMN_RENAME = OrderedDict(
        CUSTOMER = {
            "order_dow"                 : "order_day_of_week",
            "days_since_prior_order"    : "days_since_last_order",
            "department"                : "product_group",
            "department_id"             : "product_group_id",
        },
    )

    VARS = OrderedDict(
        Customer = [
            dict(var="order_id",                type="int64",   min=1,      max=206209 ,    impute="",  predictive=True ),
            dict(var="add_to_cart_order",       type="int64",   min=1,      max=145    ,    impute="",  predictive=True ),
            dict(var="reordered",               type="int64",   min=0,      max=1      ,    impute="",  predictive=True ),
            dict(var="user_id",                 type="int64",   min=0,      max=3421083,    impute="",  predictive=True ),
            dict(var="eval_set",                type="object",  min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="order_number",            type="int64",   min=1,      max=99     ,    impute="",  predictive=True ),
            dict(var="order_day_of_week",       type="int64",   min=0,      max=6      ,    impute="",  predictive=True ),
            dict(var="order_hour_of_day",       type="int64",   min=0,      max=23     ,    impute="",  predictive=True ),
            dict(var="days_since_last_order",   type="float64", min=0.0,    max=30.0   ,    impute="",  predictive=True ),
            dict(var="product_id",              type="int64",   min=1,      max=49688  ,    impute="",  predictive=True ),
            dict(var="product_name",            type="object",  min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="aisle_id",                type="int64",   min=1,      max=134    ,    impute="",  predictive=True ),
            dict(var="aisle",                   type="object",  min=None,   max=None   ,    impute="",  predictive=True ),
            dict(var="product_group_id",        type="int64",   min=1,      max=21     ,    impute="",  predictive=True ),
            dict(var="product_group",           type="object",  min=None,   max=None   ,    impute="",  predictive=True ),        
        ],

        features = [
            dict(var="num_products",        type="int64",   min=None,   max=None,   impute="",  predictive=True ),
            dict(var="peak_day",            type="int64",   min=None,   max=None,   impute="",  predictive=True ),
            dict(var="peak_time",           type="int64",   min=None,   max=None,   impute="",  predictive=True ),
            dict(var="num_orders",          type="int64",   min=None,   max=None,   impute="",  predictive=True ),
            dict(var="peakday_rate",        type="float64", min=None,   max=None,   impute="",  predictive=True ),
            dict(var="median_hour",         type="float64", min=None,   max=None,   impute="",  predictive=True ),
            dict(var="peaktime_rate",       type="float64", min=None,   max=None,   impute="",  predictive=True ),
            dict(var="mean_lag_days",       type="float64", min=None,   max=None,   impute="",  predictive=True ),
            dict(var="mean_num_products",   type="float64", min=None,   max=None,   impute="",  predictive=True ),

        ],
    )