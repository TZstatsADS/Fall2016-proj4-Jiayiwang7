from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.externals import joblib


def multi_gb(feature):    
    
    x = feature
    test_words = pd.read_csv('/Users/dd/Desktop/Columbia/Applied Data Science/Project4/lyr.csv')
    test_w = test_words.ix[:,'blank_':]
    test_w.index = test_words['dat2$track_id'].tolist()
    y = test_w
    clf = MultiOutputRegressor(GradientBoostingRegressor(random_state = 7)).fit(x, y)
    joblib.dump(clf, 'mul.pkl') 
    return clf


def multi_rf(feature, n_t, n_d):
    
    x = feature
    test_words = pd.read_csv('lyr.csv')
    test_w = test_words.ix[:,'blank_':]
    test_w.index = test_words['dat2$track_id'].tolist()
    y = test_w
    clf = MultiOutputRegressor(RandomForestRegressor(n_estimators = n_t, max_depth = n_d, random_state = 7)).fit(x, y)
    joblib.dump(clf, 'mul.pkl') 

