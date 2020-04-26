#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import pandas as pd
import  numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("train_classification.csv")
df_ce=deepcopy(df)
for name in ["Name","Sex","Ticket","Fare","Cabin","Embarked"]:
    col=df_ce[name]
    col[~col.isna()]=LabelEncoder().fit_transform(col[~col.isna()])

from missingpy import MissForest

imputer=MissForest()
imputer.fit_transform(df_ce.values.astype("float"))
