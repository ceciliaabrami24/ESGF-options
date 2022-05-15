from datetime import timedelta
from datetime import datetime
import numpy as np


def build_training_point(data, date, history_days=64, horizon_days=1):
    '''

    :param data:
    :param t_str:
    :param history_days:
    :param horizon_days:
    :return:
    '''

    # Create training example (x,y)
    try:
        x = data[(data.Date >= date - timedelta(days=(history_days - 7))) &  (data.Date <= date)]
        y = data[(data.Date >= date + timedelta(days=1)) & (data.Date <= date + timedelta(days=horizon_days))].Prix_VIX
    except KeyError:
        raise KeyError("The date {} is not in the data".format(t_str))
    # Return
    return x, y


def create_training_points(data, history_days=64, horizon_days=32):
    '''

    :param data:
    :param history_days:
    :param horizon_days:
    :return:
    '''
    X = []
    Y = []
    for t in data.index[history_days:(len(data) - horizon_days)]:
        try:
            x, y = build_training_point(data, data.Date[t], history_days=history_days, horizon_days=horizon_days)
            x.set_index('Date', inplace=True)
            X.append(x)
            Y.append(y)
        except KeyError:
            continue
    
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y


