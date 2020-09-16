# Customer_Segmentation
<p align="center"><img width="1000" height="300" src="https://miro.medium.com/max/1160/1*yf7Bk7LpZCH5wcIGSxBqjA.png"></p>

This project initiated from **Instacart** providing e-commerce shopping, with the big plan on creating a delightful shopping experience. With this transaction data with <font color='blue'>3 Million Instacart Orders</font>, we would crunch the data and extract right insight on customers, so that we can build a better shopping experience for customers in **Instacart**.

[Instacart Kaggle Challenges](https://www.kaggle.com/c/instacart-market-basket-analysis)

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Open your Jupyter notebook**


## Structuring a repository
An integral part of having reusable code is having a sensible repository structure. That is, which files do we have and how do we organise them.
- Folder layout:
```bash
customer_segmentation
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       └── index.rst
├── src
│   └── analysis
│       └── __init__.py
|       └── feature_engineer.py
|       └── statistical_analysis.py
|   └── train
│       └── __init__.py
|       └── Model.py
|   └── Config.py
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── tox.ini
```


## 1. About the Project
With this transaction data with <font color='blue'>3 Million Instacart Orders</font>, let kick started on it:
  - <b><u>RFM (Recency, Frequency, Monetory)</u></b>
  - <b><u>Market Basket Analysis</u></b>
  - <b><u>Association Rule</u></b>
  - <b><u>Customer Segmentation (unsupervised learning)</u></b>
  - <b><u>Prediction of Next Product on User will Buy</u></b>
  

## 2. Getting Started
- Prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python),
  1. Clone the repository locally
    In your terminal, use `git` to clone the repository locally.
    
    ```bash
    https://github.com/KewJS/Customer_Segmentation.git
    ```
    
    Alternatively, you can download the zip file of the repository at the top of the main page of the repository. 
    If you prefer not to use git or don't have experience with it, this a good option.
    
- Prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or


## 3. Set up your environment

### 3a. `conda` users

If this is the first time you're setting up your compute environment, 
use the `conda` package manager 
to **install all the necessary packages** 
from the provided `environment.yml` file.

```bash
conda env create -f environment.yml
```

To **activate the environment**, use the `conda activate` command.

```bash
conda activate customer_segmentation
```

**If you get an error activating the environment**, use the older `source activate` command.

```bash
source activate customer_segmentation
```

To **update the environment** based on the `environment.yml` specification file, use the `conda update` command.

```bash
conda env update -f environment.yml
```

### 3b. `pip` users

Please install all of the packages listed in the `requirement.txt`. 
An example command would be:

```bash
pip install -r requirement.txt
```


## 4. Open your Jupyter notebook

1. You will have to install a new IPython kernelspec if you created a new conda environment with `environment.yml`.
    
    ```python
    python -m ipykernel install --user --name customer_segmentation --display-name "customer_segmentation"
    ```

You can change the `--display-name` to anything you want, though if you leave it out, the kernel's display name will default to the value passed to the `--name` flag.

2. In the terminal, execute `jupyter notebook`.

Navigate to the notebooks directory and open notebook:
  - ETL: `Analysis.ipynb`
  - Modelling: `Train.ipynb`
