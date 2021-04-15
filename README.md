# Data and code repo for paper Analyzing the Impact of Missing Values and Selection Bias on Fairness

The framework of our code uses code from another Github project at https://github.com/IBM/AIF360 with some modifications. 

This repository contains the following files: <br>
<UL>
<LI>MAR_compas.py: accuracy and fairness measure before and after reweighting with MAR missing values using COMPAS data <br>
<LI>MAR_adult.py: accuracy and fairness measure before and after reweighting with MAR missing values using Adult data <br>
<LI>MNAR_compas.py: accuracy and fairness measure before and after reweighting with MNAR missing values using COMPAS data <br>
<LI>MNAR_adult.py: accuracy and fairness measure before and after reweighting with MNAR missing values using Adult data <br>
<LI>selection_compas_before_fix.py: accuracy and fairness measure before resampling with selection bias using COMPAS data <br>
<LI>selection_compas_after_fix.py: accuracy and fairness measure after resampling with selection bias using COMPAS data <br>
<LI>selection_adult_before_fix.py: accuracy and fairness measure before resampling with selection bias using Adult data <br>
<LI>selection_adult_after_fix.py: accuracy and fairness measure after resampling with selection bias using Adult data <br>
<LI>comb_compas_before_fix.py: accuracy and fairness measure before using fixing algorithms with both selection bias and missing values using COMPAS data <br>
<LI>comb_compas_stratified_resample.py: accuracy and fairness measure after using stratified resampling and reweighting with both selection bias and missing values (MAR) using COMPAS data <br>
<LI>comb_compas_unif_resample.py: accuracy and fairness measure after using uniform resampling and reweighting with both selection bias and missing values (MNAR) using COMPAS data <br>
</UL>

# Usage
This code relies on a short list of python packages, and comes with a virtual environment with the packages pre-installed.  To use it, from the root directory, run `$ source env/bin/activate` then run `pip install -r requirements.txt` to install other dependencies. <br>
If you wish to use your own environment, you need to have Python 3.7 and run `pip install -r requirements.txt` from the root directory. <br>
To run the code, run: python3 xxxxx.py

# Synthetic data
We have created a Jupyer notebook at Synthetic_data.ipynb with detailed steps of how we create the synthetic data using the method presented in the paper. 

# Reference
Rachel K. E. Bellamy Kuntal Dey and Michael Hind and Samuel C. Hoffman and Stephanie Houde and Kalapriya Kannan and Pranay Lohia and Jacquelyn Martino and Sameep Mehta and Aleksandra Mojsilovic and Seema Nagar and Karthikeyan Natesan Ramamurthy and John Richards and Diptikalyan Saha and Prasanna Sattigeri and Moninder Singh and Kush R. Varshney and Yunfeng Zhang: AI Fairness 360:  An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias (2018) https://arxiv.org/abs/1810.01943
