import math
import numpy as np
from numpy.random import randn
from numpy import exp
import pandas as pd
import datetime as dt
from itertools import repeat
from collections import OrderedDict
from IPython.display import display, Markdown, HTML

import matplotlib
import matplotlib.pyplot as plt
from termcolor import colored
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')

import scipy.stats as stats
import scipy.optimize
import scipy.spatial
from scipy.linalg import toeplitz
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp, chi2, chi2_contingency, t, sem, rankdata, norm, kurtosis
from scipy.stats import shapiro, boxcox, levene, bartlett

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot

from src.Config import Config

class Logger(object):
    info        = print
    critical    = print
    error       = print
    warning     = print
    debug       = print


class Statistic_Analysis(Config):
    def __init__(self, strings=None, suffix='', logger=Logger(), y_var='Sales'):
        self.logger = logger
        self.suffix = suffix
        self.strings = strings
        self.y_var = y_var


    @staticmethod
    def _kurt(x, normal=True):
        """Compute the kurtosis

        The kurtosis for a normal distribution is 3. For this reason, some sources use the following  
        definition of kurtosis (often referred to as "excess kurtosis"):

        Kurtosis is typically defined as:

        .. math::
            Kurt(x_0, \cdots, x_{n-1}) = \large{\frac{1}{n} \sum^{n-1}_{j=0} \large[\frac{x_j - \bar{x}}{\sigma}
            \large]^4 \large} - 3

        The :math:`-3` term is applied so a normal distribution will have a 0 kurtosis value (mesokurtic).  

        Positive kurtosis indicates a "positively skewed" or "skewed right" and negative kurtosis indicates a "negatively skewed" or "skewed left".

        Parameters
        ----------
        x : array-like
            One or two-dimensional array of data.
        normal : boolean
            Applying the data is normal distributed.

        Example
        -------
        >>> kurtosis([5, 2, 4, 5, 6, 2, 3])
        -1.4515532544378704

        Returns
        -------
        kurt : float
            If kurt = 3, normally distributed
            If kurt > 3, "positively skewed" or "skewed right"
            If kurt < 0, "negatively skewed" or "skewed left"
        """
        n = x.shape[0]
        m = np.mean(x)

        kurt = np.sum(((x-m)**4.0 / n) / np.sqrt(np.var(x))**4.0) - (3.0 * normal)

        return kurt

    
    @staticmethod
    def chi_summary(description, alpha, var1, var2, contingency, dof, chi_statistic, p_value, summary):
        test_results = {
            "Test Description"  : description,
            "Alpha"             : alpha,
            "Variable 1"        : var1,
            "Variable 2"        : var2,
            "Contingency Table" : contingency,
            "Degree of Freedom" : dof,
            "Chi_Statistic"     : chi_statistic,
            "P-Value"           : p_value,
            "Summary"           : summary
        } 
        return test_results

    
    @staticmethod
    def ttest_summary(description, alpha, sample1, sample2, population, variance1, variance2, t_statistic, p_value, summary):
        test_results = {
            "Test Description"  : description,
            "Alpha"             : alpha,
            "Sample 1 Mean"     : sample1,
            "Sample 2 Mean"     : sample2,
            "Population Mean"   : population,
            "Sample 1 Variance" : variance1,
            "Sample 2 Variance" : variance2,
            "T_Statistic"       : t_statistic,
            "P-Value"           : p_value,
            "Summary"           : summary
        } 
        return test_results

    
    @staticmethod
    def anova_table(aov):
        """Create `\eta^2` and `\omega^2` in ANOVA table

        ANOVA table provides all the information one needs in order to interprete if the results are significant. 
        However, it does not provide any effect size measures to tell if the statistical significance is meaningful.

        `\eta^2` is the exact same thing as `R^2`, except when coming from the ANOVA framework, people call it `\eta^2`.
        `\omega^2` is considered a better measure of effect size since it is unbiased in it's calculation by accounting for the degrees of freedom in the model.

        Args:
            aov (object): ANOVA table from OLS

        Returns:
            object: ANOVA table with `\eta^2` and `\omega^2` features
        """
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov

    
    def shapiro_wilk_test(self, data, var=None):
        """Check for normal distribution between groups

        We evaluate the normality of data using inference method:
            - Inference Method: Shapiro-Wilk test

        Shapiro-Wilk test evaluates a data sample and quantifies how likely the data was drawn from Gaussian Distribution.  
        The test gives us a \code

        Shapiro-Wilk test is typically defined as ``W`` value, where small value indicates that our sample is not normally distributed  
        (rejecting our null hypothesis). ``W`` is defined as:

        .. math::
            W = \frac{(\sum_{i=1}^n a_i x_(i))^2}{\sum_{i=1}^n (x_i-\bar{x})^2}

        where:   
        :math:`x_i` term is the ordered random sample values  
        :math:`a_i` term is the constant generated from the covariances, variances and means of the sample size (size, :math:`n`) from a normally distributed sample

        Null & Alternate hypothesis:
            - :math:`H_0`: Samples are normally distributed
            - :math:`H_1`: Samples are non-normally distributed

        Parameters
        ----------
        data : object
            Dataframe that has the interested column to be performed statistical analysis.
        var : array-like, optional (default=None)
            Column from the dataframe to be performed Shapiro-Wilk test. 
            If the input **data** is an array-like object, leave the option default (None).

        Example
        -------
        The way to perform normality test. We pass in an array from a datafrme based on the interested column to test on normality test.

        >>> professor_salary = [139750, 173200, 79750, 11500, 141500,
        ...                     103450, 124750, 137000, 89565, 102580]
        >>> wtest, p_value = shapiro_wilk_test(professor_salary)
        >>> wtest = 0.0869
        >>> p_value = 0.934
        >>> Sample data does not look Gaussian (fail to reject H0)
d
        Returns
        -------
        wtest: float
            W-statistics value from Shapiro-Wilk test
        p_value: float
            P-value for the test
        """
        if var != None:
            sample_data = data[var]
        else:
            sample_data = data
            
        wtest, p_value = shapiro(sample_data)
        if p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
            info = "Sample looks Gaussian (fail to reject H0)"
        else:
            info = "Sample does not look Gaussian (reject H0)"

        sample_statistics = {
                "Test Description"      : "Shapiro-Wilk Test",
                "P-Value"               : p_value,
                "Levene's Statistic"    : wtest, 
                "Test Results"          : info
            }
        return sample_statistics

    
    def levene_test(self, data, center=Config.ANALYSIS_CONFIG["LEVENE_DISTRIBUTION"], var=None):
        """Check for homogeneity of variance between groups

        Levene's test is a statistical procedure for testing equality of variances (also sometimes called homoscedasticity or homogeneity of variances)  
        between two or more sample populations.

        Levene's test is typically defined as ``W`` value, where small value indicates that at least one sample has different variance compared to population  
        (rejecting our null hypothesis). ``W`` is defined as:

        .. math::
            W = \frac{(N-k)}{(k-1)} \frac{\sum_{i=1}^k n_i(Z_i - Z_..)^2}{\sum_{i=1}^k \sum_{j=1}^{n_i} (Z_{ij} - Z_i.)^2}

        where:   
        :math:`k` term is the number of groups
        :math:`n_i` term is the number of samples belonging to the :math:`i-th` group
        :math:`N` term is the total number of samples
        :math:`Y_{ij}` term is the :math:`j-th` observation from the :math:`i-th` group

        Null & Alternative hypothesis:
            - :math:`H_0`: All of the :math:`k` sample populations have equal variances
            - :math:`H_1`: At least one of the :math:`k` sample population variances are not equal

        Parameters
        ----------
        data : object
            Dataframe that has the interested column to be performed statistical analysis.
        center : : {‘mean’, ‘median’, ‘trimmed’}, optional
            Which function of the data to use in the test. The default is ‘median’.
                - 'median' : Recommended for skewed (non-normal) distributions.
                - 'mean' : : Recommended for symmetric, moderate-tailed distributions.
                - 'trimmed' : Recommended for heavy-tailed distributions.
        var : array-like, optional (default=None)
            The sample data, possibly with different lengths.
            If the input **data** is an array-like object, leave the option default (None).

        Example
        -------
        The way to perform homogeneity of variance test. We pass in an array from a datafrme based on the interested column to test on homogeneity of variance.

        >>> col1, col2, col3 = list(range(1, 100)), list(range(50, 78)), list(range(115, 139))
        >>> wtest, p_value = levene_test(col1, col2,col3, center="mean")
        >>> wtest = 0.0869
        >>> p_value = 0.934
        >>> Sample data does not look Gaussian (fail to reject H0)

        Returns
        -------
        wtest: float
            W-statistics value from Levene's test
        p_value: float
            P-value for the test
        """
        if var != None:
            sample_data = data[var]
        else:
            sample_data = data
            
        wtest, p_value = levene(sample_data, center=center)
        if p_value > self.ANALYSIS_CONFIG["TEST_ALPHA"]:
            info = "Samples have equal variance (fail to reject H0)"
        else:
            info = "At least one of the sample has different variance from the rest (reject H0)"

        sample_statistics = {
                "Test Description"      : "Levene's Test",
                "P-Value"               : p_value,
                "Levene's Statistic"    : wtest, 
                "Test Results"          : info
            }
        return sample_statistics

    
    def bartlett_test(self, data, var=None):
        """Check for homogeneity of variance between groups, aside from Levene's test

        Bartlett's test, developed by Maurice Stevenson Bartlett, is a statistical procedure for testing if :math:`k` population samples have equal variances.

        In general, Levene's test would be prefer as it is less sensitive to non-normal samples, comparing to Barlett's test. A keynote on words *homoscedasticity*,
        which is also known as homogeneity of variances.

        Barlett's test is typically defined as :math:`X^2` value, where small value indicates that at least one sample has different variance compared to population  
        (rejecting our null hypothesis).:math:`X^2` is defined as:

        .. math::
            X^2 = \frac{(N-k)(\ln(S_{p}^2)) - \sum_{i=1}^k (N_i-1)(\ln(S_{i}^2)}{1 + (1/(3(k-1)))((\sum_{i=1}^k (1/(N_i-1)) - 1/(N-k))}

        where:   
        :math:`S_{i}^2` term is the variance of the ith groups
        :math:`N` term is the total sample size
        :math:`N_i` term is the sample size of the :math:`i-th` group
        :math:`k` term is the number of groups
        :math:`S_{p}^2` term is the pooled variance; :math:`S_{p}^2 = \sum_{i=1}^k(N_i-1)s_{i}^2 / (N-k)`

        Null & Alternative hypothesis:
            - :math:`H_0`: All of the :math:`k` sample populations have equal variances
            - :math:`H_1`: At least one of the :math:`k` sample population variances are not equal

        Parameters
        ----------
        data : object
            Dataframe that has the interested column to be performed statistical analysis.
        center : : {‘mean’, ‘median’, ‘trimmed’}, optional
            Which function of the data to use in the test. The default is ‘median’.
                - 'median' : Recommended for skewed (non-normal) distributions.
                - 'mean' : : Recommended for symmetric, moderate-tailed distributions.
                - 'trimmed' : Recommended for heavy-tailed distributions.
        var : array-like, optional (default=None)
            The sample data, possibly with different lengths.
            If the input **data** is an array-like object, leave the option default (None).

        Example
        -------
        The way to perform homogeneity of variance test. We pass in an array from a datafrme based on the interested column to test on homogeneity of variance.

        >>> col1, col2, col3 = list(range(1, 100)), list(range(50, 78)), list(range(115, 139))
        >>> wtest, p_value = bartlett_test(col1, col2, col3)
        >>> wtest = 0.0869
        >>> p_value = 0.934
        >>> Sample data does not look Gaussian (fail to reject H0)

        Returns
        -------
        wtest: float
            X^2-statistics value from Bartlett's test
        p_value: float
            P-value for the test
        """
        if var != None:
            sample_data = data[var]
        else:
            sample_data = data
            
        wtest, p_value = bartlett(sample_data)
        if p_value > self.ANALYSIS_CONFIG["TEST_ALPHA"]:
            info = "Samples have equal variance (fail to reject H0)"
        else:
            info = "At least one of the sample has different variance from the rest (reject H0)"

        sample_statistics = {
                "Test Description"      : "Bartlett's Test",
                "P-Value"               : p_value,
                "Levene's Statistic"    : wtest, 
                "Test Results"          : info
            }
        return sample_statistics


    def t_test(self, y1, y2=None, var=None, population=True, paired=False, alpha=0.05):  
        if paired and y2 is None:
            raise ValueError("Second sample is missing for paired test")
            
        if y2 is None and population is True:
            sample_1            = y1.sample(1000)
            test_description    = "One-Sample T-Test"
            
            s1_stat, s1_p_value = shapiro(sample_1[var])
            if s1_p_value > alpha:
                y1_value        = sample_1[var]
                variance1       = "Sample 1 looks Gaussian (box-cox transformation is not performed)"
            else:
                y1_value        = boxcox(sample_1[var], 0)
                variance1       = "Sample 1 do not looks Gaussian (box-cox transformation is performed)"
                
            y2_value            = None
            variance2           = None
            population_value    = y1[var].mean()
            t_test, p_value     = ttest_1samp(y1_value, population_value)
            interpretation      = f"Reject null hypothesis as p_value ({p_value}) < {alpha}" if p_value < 0.05 else f"Accept null hypothesis as p_value ({p_value}) >= {alpha}"
            test_results        = self.ttest_summary(description=test_description, 
                                                     alpha=alpha, 
                                                     sample1=y1_value.mean(), 
                                                     sample2=y2_value, 
                                                     population=population_value, 
                                                     variance1=variance1,
                                                     variance2=variance2,
                                                     t_statistic=t_test, 
                                                     p_value=p_value, 
                                                     summary=interpretation)
            
        elif (y2 is not None and var is not None) and paired == False:
            sample_1            = y1.sample(1000)
            sample_2            = y2.sample(1000)
            test_description    = "Independent Samples T-Test"
            
            s1_stat, s1_p_value = shapiro(sample_1[var])
            if s1_p_value > alpha:
                y1_value        = sample_1[var]
                variance1       = "Sample 1 looks Gaussian (box-cox transformation is not performed)"
            else:
                y1_value        = boxcox(sample_1[var], 0)
                variance1       = "Sample 1 do not looks Gaussian (box-cox transformation is performed)"
                
            s2_stat, s2_p_value = shapiro(sample_2[var])
            if s2_p_value > alpha:
                y2_value        = sample_2[var]
                variance2       = "Sample 2 looks Gaussian (box-cox transformation is not performed)"
            else:
                y2_value        = boxcox(sample_2[var], 0)
                variance2       = "Sample 2 do not looks Gaussian (box-cox transformation is performed)"
        
            population_value    = None
            t_test, p_value     = ttest_ind(y1_value, y2_value)
            interpretation      = f"Reject null hypothesis as p_value ({p_value}) < {alpha}" if p_value < 0.05 else f"Accept null hypothesis as p_value ({p_value}) >= {alpha}"
            test_results        = self.ttest_summary(description=test_description, 
                                                     alpha=alpha, 
                                                     sample1=y1_value.mean(), 
                                                     sample2=y2_value.mean(), 
                                                     population=population_value, 
                                                     variance1=variance1,
                                                     variance2=variance2,
                                                     t_statistic=t_test, 
                                                     p_value=p_value, 
                                                     summary=interpretation)
            
        elif (y2 is not None and var is not None) and paired == True:
            sample_1            = y1.sample(1000)
            sample_2            = y2.sample(1000)
            test_description    = "Paired Dependent Samples T-Test"
            
            s1_stat, s1_p_value = shapiro(sample_1[var])
            if s1_p_value > alpha:
                y1_value        = sample_1[var]
                variance1       = "Sample 1 looks Gaussian (box-cox transformation is not performed)"
            else:
                y1_value        = boxcox(sample_1[var], 0)
                variance1       = "Sample 1 do not looks Gaussian (box-cox transformation is performed)"
                
            s2_stat, s2_p_value = shapiro(sample_2[var])
            if s2_p_value > alpha:
                y2_value        = sample_2[var]
                variance2       = "Sample 2 looks Gaussian (box-cox transformation is not performed)"
            else:
                y2_value        = boxcox(sample_2[var], 0)
                variance2       = "Sample 2 do not looks Gaussian (box-cox transformation is performed)"
        
            population_value    = None
            t_test, p_value     = ttest_rel(y1_value, y2_value)
            interpretation      = f"Reject null hypothesis as p_value ({p_value}) < {alpha}" if p_value < 0.05 else f"Accept null hypothesis as p_value ({p_value}) >= {alpha}"
            test_results        = self.ttest_summary(description=test_description, 
                                                     alpha=alpha, 
                                                     sample1=y1_value.mean(), 
                                                     sample2=y2_value.mean(), 
                                                     population=population_value, 
                                                     variance1=variance1,
                                                     variance2=variance2,
                                                     t_statistic=t_test, 
                                                     p_value=p_value, 
                                                     summary=interpretation)

        else:
            self.logger.info("Failed to run test, please validate the input arguments")
            
        return test_results


    def anova_test(self, df, var1, cat_var1, cat_var2=None, two_way=False, alpha=0.05):  
        if cat_var2 is None and two_way is True:
            raise ValueError("Second variable is missing for 2-way ANOVA test")
            
        if cat_var2 is None and two_way is False:
            sample_df                   = df.sample(1000)
            test_description            = "One-Sample T-Test"
            model                       = ols(f'{var1} ~ C({cat_var1})', data=sample_df).fit()
            aov_table                   = sm.stats.anova_lm(model, typ=2)
            aov_table                   = self.anova_table(aov_table)
            aov_table['description']    = test_description
            p_value                     = aov_table['PR(>F)'][0]
            interpretation = []
            for row in aov_table.index:
                if row == f'C({cat_var1})': 
                    interpretation.append(f"Reject null hypothesis as p_value ({p_value}) < {alpha}" if p_value < 0.05 else f"Accept null hypothesis as p_value ({p_value}) >= {alpha}")
                else:
                    interpretation.append(np.nan)
            aov_table['interpretation'] = interpretation
            
        elif cat_var2 is not None and two_way is True:
            sample_df                   = df.sample(1000)
            test_description            = "Two-Sample T-Test"
            model                       = ols(f'{var1} ~ C({cat_var1}) + C({cat_var2}) + C({cat_var1}):C({cat_var2})', data=sample_df).fit()
            aov_table                   = sm.stats.anova_lm(model, typ=2)
            aov_table['description']    = test_description
            p_value_1                   = aov_table['PR(>F)'][0]
            p_value_2                   = aov_table['PR(>F)'][1]
            p_value_3                   = aov_table['PR(>F)'][2]
            interpretation = []
            for row in aov_table.index:
                if row == f'C({cat_var1})': 
                    interpretation.append(f"Reject null hypothesis as p_value ({p_value_1}) < {alpha}" if p_value_1 < 0.05 else f"Accept null hypothesis as p_value ({p_value_1}) >= {alpha}")
                elif row == f'C({cat_var2})':  
                    interpretation.append(f"Reject null hypothesis as p_value ({p_value_2}) < {alpha}" if p_value_2 < 0.05 else f"Accept null hypothesis as p_value ({p_value_2}) >= {alpha}")
                elif row == f'C({cat_var1}):C({cat_var2})':  
                    interpretation.append(f"Reject null hypothesis as p_value ({p_value_3}) < {alpha}" if p_value_3 < 0.05 else f"Accept null hypothesis as p_value ({p_value_3}) >= {alpha}")
                else:           
                    interpretation.append(np.nan)
            aov_table['interpretation'] = interpretation
            
        else:
            logger.info("Failed to run test, please validate the input arguments")
            
        return aov_table

    
    def chi_squared_test(self, df, var1, var2, alpha=Config.ANALYSIS_CONFIG["TEST_ALPHA"]):
        """Performs the Chi-square test of independence of variables

        Chi-Squared is to study the relationship between 2 categorical variables, to check is there any relationship between them.  
        In statistic, there are 2 types of variables, numerical (countable) variables and non-numerical variables (categorical) variables.  
        The Chi-Square statistic is a single number that tells you how much difference exists between our observed counts and  
        the counts we would expect if there were no relationship at all in the population. 

        Chi-Squared statistic used in Chi-Squared test is defined as:

        .. math::
            x^2_c = \sum\frac{(O_i - E_i)^2}{E_i}

        where:
        :math:`c` term is the degree of freedom
        :math:`O` term is the observed value
        :math:`E` expected value

        Null & Alternative hypothesis:
            - :math:`H_0`: There are no relationship between 2 categorical samples
            - :math:`H_1`: There is a relationship presence between 2 categorical samples

        Parameters
        ----------
        df : object
            Dataframe that contain the categorical variables
        var1 : array-like
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values.
        var2 : array-like, optional
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values.
        alpha : float
            Critical value for Chi-Squared test. The value can be found in Config file.

        Examples
        --------
        The first is to pass a dataframe with 2 different categorical group vector .
        The data used in this example is a subset of the data in Sand Advisor project on SAND_COUNT & WC.
        >>> chi_statistic = chi_squared(model_df, 'SAND_COUNT_CLASS', 'WC_CLASS')
        >>> print(chi_statistic)
            {'Test Description': 'Chi-Squared Test',
            'P-Value': 0.00033203456800745546,
            'T-Statistic': 20.896189593657517,
            'Test Results': 'Reject null hypothesis that there are no relationship between the categorical variables ...}

        Returns
        -------
        test_results : dict
            Dictionary contains the statistical analysis on chi-squared tests.
        """
        if var2 is None:
            raise ValueError("Chi Squared test require 2 categorical samples")
            
        if var1 is not None and var2 is not None:
            sample                       = df.sample(self.ANALYSIS_CONFIG["SAMPLE_SIZE"])
            test_description             = "Chi-Squared Test"
            count_data                   = pd.crosstab(sample[var1], sample[var2])
            observed_values              = count_data.values          
            stat, p_value, dof, expected = stats.chi2_contingency(count_data)
            interpretation               = f"Reject null hypothesis as p_value ({p_value}) < {alpha}" if p_value < 0.05 else f"Accept null hypothesis as p_value ({p_value}) >= {alpha}"
            test_results                 = self.chi_summary(description=test_description, 
                                                            alpha=alpha, 
                                                            var1=var1, 
                                                            var2=var2, 
                                                            contingency=count_data, 
                                                            dof=dof,
                                                            chi_statistic=stat, 
                                                            p_value=p_value, 
                                                            summary=interpretation)
        else:
            self.logger.info("Failed to run test, please validate the input arguments")
            
        return test_results

    
    def qq_quantile_plot(self, data, var, title):
        """Check for whether samples used in parametric test are in normally distributed using graphical method

        We evaluate the normality of data using inference method:
            - Graphical Method: Q-Q quantile plot

        Q-Q quantile plot is a graphical technique for determining if two datasets come from sample populatons with a common distribution (normally distributed).  
        The idealized samples are divided into groups called quantiles. Each data points in the sample is paired with a similar member from the idealized ditribution  
        at the sample cumulative distribution. 

        A perfect match for the ditribution will be shown by a line of dots on a 45-degree anfle from the bottom left of the plot to the top right.  
        Deviation by the dots from the line shows a deviation from the expected distribution.

        Parameters
        ----------
        data : object

        Retuns
        ------
        fig : QQ plot 
        """
        sample_data = data[var]

        fig, ax = plt.subplots(figsize=(8,6))
        fig = qqplot(sample_data, line="s")
        plt.title(title, weight="bold")
        plt.show()
        return fig

    
    def dist_plot_2_vars(self, df, var, title, log=False, label1=None, label2=None):
        fig, ax = plt.subplots(figsize=(20,6))
        plt.suptitle(title, fontsize=18, weight='bold')

        if log:
            var1_log = [np.log(x) for x in df[df[var] == 1]['count']]
            var2_log = [np.log(x) for x in df[df[var] == 0]['count']]
        else:
            var1_log = df[df[var] == 1]['count']
            var2_log = df[df[var] == 0]['count']
            plt.axvline(x=var1_log.mean(), label=f'{label1} Mean', color='orange', linestyle='--')
            plt.axvline(x=var2_log.mean(), label=f'{label2} Mean', color='blue', linestyle='--')
            plt.title('Mean of 1st Cond.: {:.2f};    Mean of 2nd Cond.: {:.2f}'.format(var1_log.mean(), var2_log.mean()))

        sns.distplot(var1_log, label=label1, color='orange')
        sns.distplot(var2_log, label=label2, color='blue')
        plt.legend()
        sns.despine()
        
        return fig

    
    def box_plot(self, df, xVar, yVar):
        """Boxplot 

        Detect the outliers of data across different groups

        Parameters
        ----------
        df : str
            Dataframe
        xvar : str
            Groups
        yvar : str
            Interested variable data 
        
        Returns
        -------
        fig : object
            Vertical boxplot chart of each groups of data
        """
        fig, ax = plt.subplots(figsize=(20,5))

        sns.boxplot(x=xVar, y=yVar, data=df, ax=ax)
        plt.title('Boxplot of {}'.format(yVar), size = 14, weight='bold')
        plt.xlabel('{}'.format(xVar), size = 12)
        plt.xticks(rotation = 90)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        return fig