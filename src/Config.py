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
        OUTPUT_PATH     = "processed_data",
        CUSTOMER_DATA   = "customer_data",
        TRAIN_DATA      = "train_data",
    )

    ANALYSIS_CONFIG = dict(
        FILE_TYPE       = "single",
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
            dict(var="user_id",                 min=1,      max=206209 ,    impute="",      predictive=False),
            dict(var="order_id",                min=0,      max=3421083,    impute="",      predictive=False),
            dict(var="product_id",              min=1,      max=49688  ,    impute="",      predictive=False),
            dict(var="add_to_cart_order",       min=1,      max=145    ,    impute="",      predictive=True ),
            dict(var="reordered",               min=0,      max=1      ,    impute="",      predictive=True ),
            dict(var="order_number",            min=1,      max=99     ,    impute="",      predictive=True ),
            dict(var="order_day_of_week",       min=0,      max=6      ,    impute="",      predictive=True ),
            dict(var="order_hour_of_day",       min=0,      max=23     ,    impute="",      predictive=True ),
            dict(var="days_since_last_order",   min=0.0,    max=30.0   ,    impute="",      predictive=True ),
            dict(var="product_name",            min=None,   max=None   ,    impute="",      predictive=True ),
            dict(var="aisle",                   min=None,   max=None   ,    impute="",      predictive=True ),
            dict(var="aisle_id",                min=1,      max=134    ,    impute="",      predictive=True ),
            dict(var="product_group",           min=None,   max=None   ,    impute="",      predictive=True ),
            dict(var="product_group_id",        min=1,      max=21     ,    impute="",      predictive=True ),
            dict(var="eval_set",                min=None,   max=None   ,    impute="",      predictive=True ),
            
        ],
    )