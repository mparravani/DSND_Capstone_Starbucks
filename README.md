# DSND_Capstone_Starbucks
Capstone Udacity project that analysis the impact of a portfolio of offers on an individuals spending. This analysis results in a Classification and Logistic Regression model that predicts the impact of any potential offer (existing or future) on any individual (existing or future).

The purpose of these models would be to feed into an optimization analysis; combining profitability of the product and offers, with customer lifetime value, and the impact of the offers to influence spending. Subsequently tailoring the product offerings to individuals (optimizing duration, difficulty, channel, etc).

## Prerequisites
The following packages are required;   
sklearn, xgboost, pandas, numpy, matplotlib, seaborn

Install using the code below  
```
!pip install sklearn, xgboost, pandas, numpy, matplotlib, seaborn
```

## Instructions

*Starbucks_Capstone_Notebook.ipynb* is the main script. It contains all detailed analysis and results.

*functions.py* contains all processing functions to clean and wrangle data, as well as build and train models.



## Datasets
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record
