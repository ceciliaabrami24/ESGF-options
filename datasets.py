import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_csv(file):
    with open("../artefacts/data/"+file, "r") as f:
        aa=f.readlines()

    aa = [a.replace("\t", ";").replace(",", ".") for a in aa]
    aa = "".join(aa)
    with open("../artefacts/data/"+file.split(".")[0]+".csv","w") as f:
        f.write(aa)


        
def load_csv(file):
    df = pd.read_csv("../artefacts/data/"+file.split(".")[0]+".csv", sep = ";", header = None)
    df.columns = ["Date", "Prix_"+file.split(".")[0]]
    return df

def concat_csv(dfs):
    series = []
    date_str = dfs[0]["Date"]
    series.append(pd.Series([datetime.strptime(date,"%d/%m/%Y" ) for date in date_str]))
    for df in dfs:
        series.append(pd.Series(df[df.columns[1]]))
    df = pd.concat(series,axis=1)
    cols = list(df.columns)
    cols[0] = "Date"
    df.columns = cols
    return df

def load_data(files):
    dfs = []
    for file in files:
        create_csv(file)
        dfs.append(load_csv(file))
    df = concat_csv(dfs)
    return df


def split_train_data(data: pd.DataFrame, split_year: int=1987) -> pd.DataFrame:
    '''
    Split the melbourne data into a training dataframe and a test dataframe.
    The training data is composed of all temperature points strictly anterior to the given split year.
    The test data is composed of all the points posterior or equal to the split year.
    :param melbourne_data: pd.DataFrame, with at least column ['Date']
    :param split_year: str, the year to split the data on
    :return: (pd.DataFrame, pd.DataFrame)
    '''

    # Format split year variable
    split_date =  datetime(split_year,1,1)
    # Trainings data. Data anterior to the given split year
    train_data = data[data.Date < split_date]

    return train_data

def create_test_data(data,  history_days = 30, horizon_days = 30,date = datetime(1987,1,1)):

        
    # Define the time window of extraction
    start_date =  date - timedelta(days = history_days)
    stop_date = date +timedelta(days = horizon_days)
    #Extract data in the time window defined
    data = data[(data.Date >= start_date) & (data.Date < stop_date)]


    #Every datapoints before January 1st of the year is a feature, and every datapoints after January first are our targets
    X = data[data.Date < date]
    Y = data[data.Date >= date].Prix_VIX
    
    X.set_index('Date', inplace=True)
    
    X = np.stack([X])
    Y = np.stack([Y])
    return X, Y




