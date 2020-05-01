#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
from time import time
df=pd.read_csv("../exmaple/train_classification.csv")
start=time()
df.pop("Name")
df.pop("Ticket")
from skimpute import MissForest
ans=MissForest(consider_ordinal_as_cat=True).fit_transform(df)
print(ans)
print(time()-start)
