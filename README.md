# Customer_Segmentation
<p align="center"><img width="1000" height="300" src="https://miro.medium.com/max/1160/1*yf7Bk7LpZCH5wcIGSxBqjA.png"></p>

## Table of Contents


## 1. About the Project
Its [Instacart Kaggle Challenges](https://www.kaggle.com/c/instacart-market-basket-analysis) coming from <b>Instacart</b>, with the big plan on creating a delightful shopping experience. With the transactional data provided, we can perform: 

  - <b><u>RFM (Recency, Frequency, Monetory)</u></b>
  - <b><u>Market Basket Analysis</u></b>
  - <b><u>Association Rule</u></b>
  - <b><u>Customer Segmentation (unsupervised learning)</u></b>
  - <b><u>Prediction of Next Product on User will Buy</u></b>
  
With this transaction data with <font color='blue'>3 Million Instacart Orders</font>, let kick started on it!
  
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
