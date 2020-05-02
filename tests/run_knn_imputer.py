#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from time import time

import pandas as pd

from skimpute import KNNImputer

df = pd.read_csv("../exmaple/train_classification.csv")
start = time()
df.pop("Name")
df.pop("Ticket")
y = df.pop("Survived").values

ans = KNNImputer().fit_transform(df, y)
print(ans)
print(ans.dtypes)
print(time() - start)
