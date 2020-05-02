#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

from skimpute import MissForest

df = pd.read_csv("../exmaple/train_classification.csv")
start = time()
df.pop("Name")
df.pop("Ticket")
df.pop("PassengerId")
y = df.pop("Survived").values
cv = ShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_ix, test_ix = next(cv.split(df, y))
train_X = (df.iloc[train_ix, :])
train_y = y[train_ix]
test_X = (df.iloc[test_ix, :])
test_y = y[test_ix]
imputer = MissForest()
imputer.fit(df)
train_X = imputer.transform(train_X)
test_X = imputer.transform(test_X)
print(train_X)
print(train_X.dtypes)
print(time() - start)
encoder = OrdinalEncoder()
encoder.fit(df)
train_X = encoder.transform(train_X)
test_X = encoder.transform(test_X)
rf = RandomForestClassifier(random_state=42)
rf.fit(train_X, train_y)
score = rf.score(test_X, test_y)
print(score)  # 0.8295964125560538
