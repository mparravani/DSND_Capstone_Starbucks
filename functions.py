import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns

def import_data (fp_portfolio, fp_profile, fp_transcript):
    '''
    Imports the datasets into data frames

    Arguments:
        fp_portfolio {string} -- file location of porfolio dataset
        fp_profile {string} -- file location of profile dataset
        fp_transcript {string} -- file location of transcript dataset

    Returns:
        portfolio {dataframe} -- [description]
        profile {dataframe} -- [description]
        transcript {dataframe} -- [description]
    '''    
    
    portfolio = pd.read_json(fp_portfolio, orient='records', lines=True)
    profile = pd.read_json(fp_profile, orient='records', lines=True)
    transcript = pd.read_json(fp_transcript, orient='records', lines=True) 

    return portfolio, profile, transcript

def clean_portfolio(portfolio):
    

    # One-Hot Encoding of Channels and Offer_Type

    # Separate channels into multiple columns
    channels = []
    for i in portfolio.channels:
        channels.extend(i)
    channels = list(set(channels))

    for channel in channels:
        portfolio[channel] = portfolio.channels.apply(lambda x: 1 if channel in x else 0)

    #separate offer_type into multiple columns
    offers = portfolio.offer_type.value_counts().index

    for offer in offers:
        portfolio[offer] = portfolio.offer_type.apply(lambda x: 1 if offer in x else 0)

    # drop redundant columns
    portfolio.drop(columns = ['offer_type','channels'], inplace = True)

    return portfolio
