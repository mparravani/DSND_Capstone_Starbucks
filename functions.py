import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
import os
import pickle as pkl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier

def import_data (fp_portfolio ='data/portfolio.json', fp_profile='data/profile.json', fp_transcript='data/transcript.json'):
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
    '''
    Cleans and one-hot-encodes portfolio datasets (separating channel and offer type into binaries)

    Arguments:
        portfolio {dataframe} -- imported porfolio dataframe of offers

    Returns:
        portfolio -- cleaned and on-hot-encoded dataset
    ''' 

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


def influenced_check(trns_row, offers): 
    """
    Establish if a transaction was made under the influence of an offer
    Used in Transaction data processing

    Arguments:
        trns_row {series} -- row of transactions dataset
        offers {dataframe} -- offers dataframe, as established in below function
        
    Returns:
        Tuple:
            Binary of whether transaction was made under the influence of an offer
            Active offer id, or np.nan where no influence
   """   
    # filter offers table for person
    person_offers = offers[offers.person == trns_row.person]

    #cycle through offers
    for _, offer_row in person_offers.iterrows():
        
        #if transaction was within span of influence
        #time viewed is not nan, transaction time is > time viewed, and trs time is < end of influence
        if (offer_row.time_viewed != np.nan) & (trns_row.time >=offer_row.time_viewed) & (trns_row.time <=offer_row.end_of_influence):
            return [True, offer_row.offer ]

    #if transaction was outside of any influence span
    return [False, np.nan]


def avg_daily_value (transactions):
    '''
    Calculates the average daily spend of customers through different time periods. Used for processing transactions dataset 

    Arguments:
        transactions {dataframe} -- partially cleaned transactions dataset  

    Returns:
        person_offer_spend {dataframe} -- average daily spend for each individual for each offer (and no-influence timeframe)
    '''    
    
    person_offer_spend = pd.DataFrame(columns = ['person','offer','avg_daily_spend', 'total_spend', 'days'])

    for person in set(transactions.person):
        
        #subset dataframe for each person
        person_trns = transactions[transactions.person==person]

        #create time period indexed column
        person_trns.sort_values(by = 'time',inplace=True)
        person_trns['time_per'] = (person_trns['offer'] != person_trns['offer'].shift(1)).cumsum()

        # instantiate previous max day indicator
        prev_max = 0
        
        # loop through time periods
        for i in set(person_trns['time_per']):

            # find the max day in this time period
            # limitation: for the last time period, it will not assume the 29th day (last day) - more logic is required
                # the case of an offer ending but no influced transactions following is missed with this. Can be added in future version
            cur_max = person_trns[person_trns.time_per==i].day.max()

            # gather variables to append to resultant dataset
            offer = person_trns[person_trns.time_per==i].iloc[0].offer
            total_spend = person_trns[person_trns.time_per==i].value.sum()
            days = cur_max-prev_max + 1
            avg_daily = total_spend/days

            # establish row and append to resultant dataset
            row = {'person':person, 'offer':offer, 'avg_daily_spend': avg_daily, 'total_spend':total_spend, 'days':days}
            person_offer_spend=person_offer_spend.append(row, ignore_index=True)

            #iterate prev_max counter for next cycle
            prev_max = cur_max

    return person_offer_spend



def calculate_offer_impact(transcript, portfolio, update = False, offer_impactfp = 'data/offer_impact.pickle'):
    '''
    Processes transcript data and merges with portfolio to establish the impact of each offer on each individual (Lift).
    Lift calculated as average spend during the promotion relative to the user's un-influenced spending. 

    Processing takes a long time, if calculations are not required update = False will return a previously calculated version.

    Arguments:
        transcript {dataframe} -- transcript dataframe as imported
        portfolio {dataframe} -- portfolio dataframe as imported
        update {bool} -- whether there has been an update to the data - if False will return previously save version (default: {False})
        offer_impactfp {str} -- file location of offer_impact dataset (default: {'data/offer_impact.pickle'})

    Returns:
        offer_impact -- dataset of ever offer provided to each individual and the impact of the offer (lift) on spending
    '''

    # check if an update to the file is called for, if not - return the previous version
    if update == False:
        print ('Importing Previously Saved Dataset')
        offer_impact = pd.read_pickle(offer_impactfp)
        return offer_impact
    else:
        
        print('Data Cleaning')
        #split 'value' column
        transcript['value_label'] = transcript.value.apply(lambda x: [*x][0])
        transcript['value'] = transcript.value.apply(lambda x: list(x.values())[0])

        #convert transaction time into days
        transcript.time = transcript.time/24

        # separate into transactions and offers
        transactions = transcript[transcript.event == 'transaction'][['person','value','time']]
        offers = transcript[transcript.event != 'transaction']

        #pivot for each person and subsequent offer
        offers = offers.pivot_table(values = 'time',index = ['person','value'], columns = 'event').reset_index()
        offers.columns = ['person','offer','time_completed', 'time_received','time_viewed']

        #merge offers dataframe with portfolio to establish timelines of influence
        offers = offers.merge(portfolio, how = 'left',left_on='offer', right_on= 'id').drop(columns='id')
        offers['end_of_influence'] = offers.apply(lambda x: min([x.time_received+x.duration, x.time_completed]),axis = 1)

        print('Establishing which transactions were influenced')
        #loop through all transactions to test if it was influenced. 
        tmp = transactions.apply(lambda x: influenced_check(x,offers), axis=1).apply(pd.Series)
        tmp.columns = ['influenced','offer']
        
        # merge back to transactions dataset and note day of occurance
        transactions = pd.concat([transactions,tmp],axis=1)
        transactions['day'] = np.floor(transactions.time)
        transactions.offer.fillna(0,inplace = True)

        print('Calculating Average Daily Totals by Time Perioed')
        # find every individual's average daily average for each time period 
        # (offer standing, in between offers, etc)
        trns_avg_daily = avg_daily_value(transactions)

        print('Wrangling and merging final dataset')
        # extract no-influene times and take weighted average
        no_influence = trns_avg_daily[trns_avg_daily.offer==0].groupby(by = 'person',as_index=False).sum()
        no_influence['avg_daily_spend']=no_influence.total_spend/no_influence.days

        no_influence.drop(columns = ['total_spend','days'],inplace = True) 
        no_influence.columns = ['person','no_influence_avg_daily_spend']
        
        #merge offer impact with no offer transaction behavior
        offer_impact = trns_avg_daily[trns_avg_daily.offer != 0][['person','offer','avg_daily_spend']]
        offer_impact.columns = ['person','offer','offer_daily_spend']
        offer_impact = offer_impact.merge(no_influence, how='inner',left_on = 'person',right_on = 'person')
        offer_impact['lift'] = offer_impact.offer_daily_spend/offer_impact.no_influence_avg_daily_spend

        print('Saving to pickle file')
        # save updated file as pickle
        offer_impact.to_pickle(offer_impactfp)
        return offer_impact


def clean_profile (profile, update = False, profile_fp = 'data/profile.pickle'):
    '''
    Cleans profile dataset and imputes missing data using KNN

    Arguments:
        profile {dataframe} -- profile dataset as imported

    Keyword Arguments:
        update {bool} -- whether there has been an update to the data - if False will return previously save version (default: {False})
        profile_fp {str} -- [file location of profile dataset (default: {'data/profile.pickle'})

    Returns:
        profile {dataframe} -- cleaned and KNN filled profile dataset
    '''
    if update == False:
        print ('Importing Previously Saved Dataset')
        profile = pd.read_pickle(profile_fp)
        return profile
    else:

        print('Data Cleaning')
        #establish NAs instead of None
        profile.gender = profile.gender.replace({'None':np.nan})
        profile.age=profile.age.replace({118:np.nan})

        # convert became member on to datetime, then establish 'months since join'
        profile.became_member_on = profile.became_member_on.apply(lambda x: pd.to_datetime(str(x),format = '%Y%m%d'))
        profile['months_since_join']=profile.became_member_on.apply(lambda x: pd.Timedelta(pd.Timestamp.today()-x).days/30)

        #convert the gender column into binary
        profile.gender=profile.gender.replace({'F':0, 'M':1, 'O':np.nan})
        profile.gender = pd.to_numeric(profile.gender)

        print('Imputing Missing Data')
        #impute missing data using KNN Imputer
        from sklearn.impute import KNNImputer
        imp = KNNImputer()
        profile_imp = pd.DataFrame(imp.fit_transform(profile[['gender','age','income','months_since_join']]), columns = ['gender','age','income','months_since_join'])

        # Join person ID back onto imputed values
        profile = pd.concat([profile_imp,profile.id],axis = 1)
        # round gender value back to 1/0
        profile.gender = round(profile.gender,0)
        
        print('Saving to pickle file')
        profile.to_pickle(profile_fp)

        return profile

def IQR_adjustment(data,colname):
    '''
    Outlier adjustment using inner quartile range (max Q3+1.5*IQR), also plotting a boxplot for the comparable distributions

    Arguments:
        data {series} -- Series to be processed
        colname {string} -- column name being processed

    Returns:
        adj_data {series} -- Adjusted series (with dampened outlers)
    '''
    # 

    # establish IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    adj_data = copy(data)
    adj_data[adj_data>(Q3+1.5*IQR)]=(Q3+1.5*IQR)
    adj_data[adj_data<(Q1-1.5*IQR)]=(Q1-1.5*IQR)
    
    fig, ax = plt.subplots(1,2, figsize =(12,4),  sharey=True)
    plt.axes(ax[0])
    sns.boxplot(data)
    plt.title('Before Adjustment')
    
    plt.axes(ax[1])
    sns.boxplot(adj_data)
    plt.title('After Adjustments')
    
    fig.suptitle('Outlier Processing for ' + colname, fontsize=16)
    plt.show()
    return adj_data


def prep_for_binary_model(offer_impact, profile, portfolio, lift_threshold = 1.25):
    '''
    Merge datasets on person id and offer id for all parameters. Then return a binary of if lift was greater than threshold.

    Arguments:
        offer_impact {dataframe} -- offer_impact dataframe as prepared above
        profile {dataframe} -- profile dataframe as prepared above
        portfolio {dataframe} -- portfolio dataframe as prepared above

    Returns:
        model_data -- resultant merged dataframe with 'binary_lift' column indicating sucessful result of spending
    '''

    # join person information and offer information to main table
    model_data = offer_impact.merge(profile, how = 'inner',left_on='person',right_on= 'id')
    model_data.drop(columns=['offer_daily_spend','person','id'], inplace = True)

    model_data = model_data.merge(portfolio, how = 'inner',left_on='offer',right_on= 'id')
    model_data.drop(columns=['offer','id'], inplace = True)

    model_data['binary_lift'] = model_data.lift>lift_threshold

    return model_data


def build_classifier (run_gridsearch):

    # build model
    xgb_classifier = XGBClassifier(learning_rate =0.1,
                        n_estimators=1000,
                        max_depth=2,
                        min_child_weight=3,
                        gamma=0.4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread=-1,
                        scale_pos_weight=1)

    params = {
    'max_depth':[2,5,7],
    'min_child_weight':[2,3,4],
    'gamma':[0.2,0.4,0.6]
    }
    
    # recent success gridsearch parameters are already used in model. If update requested, re-run gridsearch
    if run_gridsearch:
        return GridSearchCV(xgb_classifier, param_grid=params,  cv=5, n_jobs=-1, scoring='roc_auc')
    else:
        return xgb_classifier


def build_train_classifier(X,y, import_model = True, run_gridsearch = False, model_fp = 'model/binary_classifier.pickle'):

    if import_model == True:
        print ('Importing Previously Saved Model')
        binary_classifier = pkl.load(open(model_fp, 'rb') ) 
        return binary_classifier
    else:

        print ('Building Model')
        binary_classifier = build_classifier(run_gridsearch)

        # prep train_test data, then train model
        print('Training Model')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        binary_classifier.fit(X_train, y_train)

        print('Evaluating on Test Data')
        # make predictions and return accuracy
        pred = binary_classifier.predict(X_test)
        accuracy = accuracy_score(y_test,pred)

        print('Saving Model')
        with open(model_fp, 'wb') as f:
            pkl.dump(binary_classifier, f)

    return binary_classifier, accuracy

def build_train_log_regression(X,y, import_model = True, run_gridsearch = False, model_fp = 'model/log_regression.pickle'):

    if import_model == True:
        print ('Importing Previously Saved Model')
        log_regression = pkl.load(open(model_fp, 'rb') ) 
        return log_regression
    else:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print ('Building Model')
        log_regression = LogisticRegression(max_iter = 100)
        log_regression.fit(X_train,y_train)
        pred = log_regression.predict(X_test)
        accuracy_score(y_test,pred)

        # prep train_test data, then train model
        print('Training Model')
        log_regression.fit(X_train, y_train)

        print('Evaluating on Test Data')
        # make predictions and return accuracy
        pred = log_regression.predict(X_test)
        accuracy = accuracy_score(y_test,pred)

        print('Saving Model')
        with open(model_fp, 'wb') as f:
            pkl.dump(log_regression, f)

    return log_regression, accuracy
