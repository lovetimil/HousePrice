#coding:utf-8
# 1- 76 direct
import cPickle
import numpy as np
np.random.seed(10)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit,KFold
import random
from curve import plot_learning_curve
#from GBDT_LR import Assembly_paralel,Assembly_cascade,score_p,score_c,
from StringTools import StringTools

from TextGrocery import TextBatch,StringT
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
class PriceModel(object):







































    def __init__(self):


        self.textmodel = TextBatch()
        self.string_tool = StringT()
        self.enc = OneHotEncoder()
        self.model = SVR(kernel='rbf',C= 0.1,gamma= 0.1)

    def train(self,file):
        (X,y) = self.preprocess(file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)
        self.model.fit(X,y)
        self.test(X,y)
        return self
    def test(self,X,y):
        y_predict = self.model.predict(X)
        print mean_squared_error(y, y_predict)

    def preprocess(self,file):
        df = pd.read_csv(file)
        interval_years = df.icol(1).map(self.string_tool.getYears()).as_matrix()
        portait = df.icol(2).map(self.string_tool.getPortaitNum()).as_matrix()
        text = df.iloc[:,[4,10,11,12]].as_matrix()

        price = df.icol(5).map(self.string_tool.getPrice()).as_matrix()
        housearea = df.icol(6).map(self.string_tool.getArea()).as_matrix()
        enum_f = df.iloc[:,[3,7,8]]
        trans_enmuf = enum_f.applymap(self.string_tool.String2id()).as_matrix()
        trans_enmuf = self.enc.fit_transform(trans_enmuf)

        landarea = df.icol(9).map(self.string_tool.getArea()).as_matrix()

        textf = self.textmodel.fit_transform(text).as_matrix()

        X = np.hstack((interval_years,portait,housearea,landarea,trans_enmuf,textf))
        y = price
        return (X,y)
    def predict(self,X):
        X = pd.DataFrame(X)
        interval_years = X.icol(1).map(self.string_tool.getYears()).as_matrix()
        portait = X.icol(2).map(self.string_tool.getPortaitNum()).as_matrix()
        text = X.iloc[:,[4,9,10,11]].as_matrix()

        price = X.icol(5).map(self.string_tool.getPrice()).as_matrix()
        housearea =X.icol(5).map(self.string_tool.getArea()).as_matrix()
        enum_f = X.iloc[:,[3,6,7]]
        trans_enmuf = enum_f.applymap(self.string_tool.String2id()).as_matrix()
        trans_enmuf = self.enc.transform(trans_enmuf)

        landarea = X.icol(8).map(self.string_tool.getArea()).as_matrix()

        textf = self.textmodel.fit(text).as_matrix()

        X = np.hstack((interval_years,portait,housearea,landarea,trans_enmuf,textf))
        ret = self.model.predict(X,y)
        return ret
    def refreshpath(self,path):
        self.textmodelpath = "%s_%s" % (path, "textmodel")
        self.encfilepath = "%s_%s" % (path, "enc")
        self.modelfilepath = "%s_%s" % (path, "model")
    def save(self,path):
        self.refreshpath(path)
        self.textmodel.save(self.textmodelpath)
        cPickle.dump(self.enc, open(self.encfilepath, 'wb'), -1)
        cPickle.dump(self.model,open(self.modelfilepath,'wb'),-1)
        return self
    def load(self,path):
        self.refreshpath(path)
        self.textmodel = self.textmodel.load(self.textmodelpath)
        self.enc = cPickle.load(open(self.encfilepath,"rb"))
        self.model = cPickle.load(open(self.modelfilepath,"rb"))
        return self




