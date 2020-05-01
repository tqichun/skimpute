#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
df=pd.read_csv("../exmaple/train_classification.csv")
from skimpute import AdaptiveSimpleImputer
ans=AdaptiveSimpleImputer(consider_ordinal_as_cat=True).fit_transform(df)
print(ans)
