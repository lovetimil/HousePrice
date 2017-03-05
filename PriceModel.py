#coding:utf-8
# 1- 76 direct
import cPickle
import numpy as np
np.random.seed(10)
import os
import shutil
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
#from curve import plot_learning_curve
#from GBDT_LR import Assembly_paralel,Assembly_cascade,score_p,score_c,
#from StringTools import StringTools

from TextGrocery import TextBatch,StringT
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import robust_scale
from sklearn.ensemble import GradientBoostingRegressor


class PriceModel(object):

    def __init__(self):


        self.textmodel = TextBatch()
        self.string_tool = StringT()
        self.enc = OneHotEncoder()
        #self.model = SVR(kernel='rbf',C= 0.1,gamma= 0.1)
        self.model = GradientBoostingRegressor()
        self.pdX = None

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
        #df = pd.read_csv(file).head(20)
        df = pd.read_csv(file)
        text = df.iloc[:,[4,10,11,12]].as_matrix()
        textf = self.textmodel.fit_transform(text)
        (a,b) = textf.shape
        #print("text shape",a,b)
        interval_years = df.icol(1).map(self.string_tool.getYears).as_matrix().reshape(a,1)
        portait = df.icol(2).map(self.string_tool.getPortaitNum).as_matrix().reshape(a,1)


        price = df.icol(5).map(self.string_tool.getPrice).as_matrix()
        housearea = df.icol(6).map(self.string_tool.getArea).as_matrix().reshape(a,1)
        enum_f = df.iloc[:,[3,7,8]]
        trans_enmuf = enum_f.applymap(self.string_tool.String2id).as_matrix()
        trans_enmuf = self.enc.fit_transform(trans_enmuf).toarray()
        #print('trans_enumn',trans_enmuf.shape)
        landarea = df.icol(9).map(self.string_tool.getArea).as_matrix().reshape(a,1)

        #textf = self.textmodel.fit_transform(text).as_matrix()
        continue_f = robust_scale(np.hstack((interval_years,portait,housearea,landarea)),1)
        #X = np.hstack((interval_years,portait,housearea,landarea,trans_enmuf,textf))
        X = np.hstack((continue_f,trans_enmuf,textf))
        self.pdX = pd.DataFrame(X).fillna(df.mean())


        X = self.pdX.as_matrix()

        y = price
        return (X,y)
    def predict(self,X):
        X = pd.DataFrame(X)
        text = X.iloc[:, [4, 9, 10, 11]].as_matrix()
        textf = self.textmodel.transform(text)
        a = 0
        try:
            (a,_) = textf.shape
        except:
            pass
        #print("text shape",a)


        interval_years = X.icol(1).map(self.string_tool.getYears).as_matrix().reshape(a,1)
        portait = X.icol(2).map(self.string_tool.getPortaitNum).as_matrix().reshape(a,1)


        #price = X.icol(5).map(self.string_tool.getPrice()).as_matrix().reshape(a,1)
        housearea =X.icol(5).map(self.string_tool.getArea).as_matrix().reshape(a,1)
        enum_f = X.iloc[:,[3,6,7]]
        trans_enmuf = enum_f.applymap(self.string_tool.String2id).as_matrix()
        trans_enmuf = self.enc.transform(trans_enmuf).toarray()
        #print('trans_enumn',trans_enmuf.shape)

        landarea = X.icol(8).map(self.string_tool.getArea).as_matrix().reshape(a,1)


        continue_f = robust_scale(np.hstack((interval_years,portait,housearea,landarea)),1)
        #X = np.hstack((interval_years,portait,housearea,landarea,trans_enmuf,textf))
        #X = np.hstack((interval_years,portait,housearea,landarea,trans_enmuf,textf))
        X = np.hstack((continue_f,trans_enmuf,textf))
        (a,b) = self.pdX.shape
        tX = pd.concat([self.pdX,pd.DataFrame(X)])
        tX = tX.fillna(tX.mean())
        X = tX.iloc[a:].as_matrix()

        ret = self.model.predict(X)
        return ret
    def refreshpath(self,path):
        self.textmodelpath = "%s_%s" % (path, "textmodel")
        self.encfilepath = "%s_%s" % (path, "enc")
        self.modelfilepath = "%s_%s" % (path, "model")
        self.pdXfilepath = "%s_%s"%(path,"pdx")
    def save(self,path):
        self.refreshpath(path)
        if os.path.exists(path) :
            shutil.rmtree(path)
        try:
            os.mkdir(path)
        except OSError as e:
            raise OSError(e, 'Please use force option to overwrite the existing files.')
        self.textmodel.save(self.textmodelpath)
        cPickle.dump(self.enc, open(self.encfilepath, 'wb'), -1)
        cPickle.dump(self.model,open(self.modelfilepath,'wb'),-1)
        cPickle.dump(self.pdX,open(self.pdXfilepath,'wb'),-1)


        return self
    def load(self,path):
        self.refreshpath(path)
        self.textmodel = self.textmodel.load(self.textmodelpath)
        self.enc = cPickle.load(open(self.encfilepath,"rb"))
        self.model = cPickle.load(open(self.modelfilepath,"rb"))
        self.pdX = cPickle.load(open(self.pdXfilepath,"rb"))
        return self




