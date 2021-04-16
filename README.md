# Data and code repo for paper Analyzing the Impact of Missing Values and Selection Bias on Fairness

The framework of our code uses code from another Github project at https://github.com/IBM/AIF360 with some modifications. 

This repository contains the following files: <br>
<UL>
<LI>demo_missing_adult.ipynb: this file provides a step by step instruction to create MAR and MNAR type missing values using Adult data to show the negative effects of MAR and MNAR missing values and show the performance of our reweighting algorithm. <br>
<LI>demo_missing_compas.ipynb: this file provides a step by step instruction to create MAR and MNAR type missing values using COMPAS data to show the negative effects of MAR and MNAR missing values and show the performance of our reweighting algorithm. <br>
<LI>demo_sel_adult.ipynb: this file provides a step by step instruction to create selection bias using Adult data to show the negative effects of selection and show the performance of our resampling algorithm. <br>
<LI>demo_sel_compas.ipynb: this file provides a step by step instruction to create selection bias using COMPAS data to show the negative effects of selection and show the performance of our resampling algorithm. <br>
<LI>demo_sel_missing.ipynb: this file provides a step by step instruction to create selection bias and MAR type missing values within the training dataset on COMPAS data. This file is a combination of demo_missing_compas.ipynb and demo_sel_compas.ipynb. Make sure you have read and understood demo_missing_compas.ipynb and demo_sel_compas.ipynb of how to create missing values and selection bias. <br>

</UL>

# Usage
This code relies on a short list of python packages, and comes with a virtual environment with the packages pre-installed.  To use it, from the root directory, run `$ source env/bin/activate` then run `pip install -r requirements.txt` to install other dependencies. <br>
If you wish to use your own environment, you need to have Python 3.7 and run `pip install -r requirements.txt` from the root directory. <br>

To run the code, run: `jupyter lab` in command line to open a jupyter lab and open the ipynb files

# Synthetic data
We have created a Jupyer notebook at Synthetic_data.ipynb with detailed steps of how we create the synthetic data using the method presented in the paper. 

# Reference
Rachel K. E. Bellamy Kuntal Dey and Michael Hind and Samuel C. Hoffman and Stephanie Houde and Kalapriya Kannan and Pranay Lohia and Jacquelyn Martino and Sameep Mehta and Aleksandra Mojsilovic and Seema Nagar and Karthikeyan Natesan Ramamurthy and John Richards and Diptikalyan Saha and Prasanna Sattigeri and Moninder Singh and Kush R. Varshney and Yunfeng Zhang: AI Fairness 360:  An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias (2018) https://arxiv.org/abs/1810.01943
