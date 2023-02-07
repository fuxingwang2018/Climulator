
# https://www.kdnuggets.com/2018/06/step-forward-feature-selection-python.html

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import sys

def stepwise_algorithm(X_train, X_test, y_train, y_test):

    X_train, X_test, y_train, y_test = get_example_data()

    K_Features = 1
    print('Training dataset shape:', X_train.shape, y_train.shape)
    print('Testing dataset shape:', X_test.shape, y_test.shape)

    #sys.exit()
    # Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Build step forward feature selection
    sfs1 = sfs(clf,
           k_features=K_Features,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

    # Perform SFFS
    sfs1 = sfs1.fit(X_train, y_train)

    # Which features?
    feat_cols = list(sfs1.k_feature_idx_)
    print(feat_cols)

    # Build full model with selected features
    clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
    clf.fit(X_train[:, feat_cols], y_train)

    y_train_pred = clf.predict(X_train[:, feat_cols])
    print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

    y_test_pred = clf.predict(X_test[:, feat_cols])
    print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

    # Build full model on ALL features, for comparison
    clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

    y_test_pred = clf.predict(X_test)
    print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))

def get_example_data():

    # Read data
    df = pd.read_csv('/nobackup/rossby26/users/sm_fuxwa/AI/winequality/winequality-white.csv', sep=';')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.values[:,:-1],
        df.values[:,-1:],
        test_size=0.25,
        random_state=42)

    print('Training dataset shape before ravel:', X_train.shape, y_train.shape)
    print('Testing dataset shape before ravel:', X_test.shape, y_test.shape)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test

