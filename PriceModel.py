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
class PriceModel(object):







































    def __init__(self):


        self.textmodel = TextBatch()
        self.string_tool = StringT()
        self.enc = OneHotEncoder()
        self.model = SVR(kernel='rbf',C= 0.1,gamma= 0.1)

    def train(self,yearf,textf,enmuf,continue_f,label):
        yearf = pd.DataFrame(yearf)
        trans_f = yearf.applymap(self.string_tool.getYears).as_matrix()
        trans_enmuf = pd.DataFrame(enmuf).applymap(self.string_tool.String2id()).as_matrix()
        textf = self.textmodel.train(textf).as_matrix()
        trans_enmuf = self.enc.fit_transform(trans_enmuf)
        X = np.hstack((yearf,trans_enmuf,continue_f))
        self.model.fit(X,y)
        return self

    def predict(self,yearf,textf,enmuf,continue_f):
        yearf = pd.DataFrame(yearf)
        trans_f = yearf.applymap(self.string_tool.getYears).as_matrix()
        trans_enmuf = pd.DataFrame(enmuf).applymap(self.string_tool.String2id()).as_matrix()
        textf = self.textmodel.fit(textf).as_matrix()
        trans_enmuf = self.enc.transform(trans_enmuf)
        X = np.hstack((yearf,trans_enmuf,continue_f))
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




