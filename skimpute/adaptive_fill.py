#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils.multiclass import type_of_target


def is_cat(s, consider_ordinal_as_cat):
    for elem in s:
        if isinstance(elem, (float, int)):
            continue
        else:
            return True
    if consider_ordinal_as_cat:
        if s.dtype == object:
            s = s.astype(float)
        tp = type_of_target(s)
        if tp in ("multiclass", "binary"):
            return True
    return False


def parse_cat_col(X_, consider_ordinal_as_cat):
    cat_idx = []
    num_idx = []
    for i in range(X_.shape[1]):
        col = X_[:, i]
        if is_cat(col, consider_ordinal_as_cat):
            cat_idx.append(i)
        else:
            num_idx.append(i)
    return np.array(num_idx), np.array(cat_idx)


class AdaptiveFill(BaseEstimator, TransformerMixin):
    def __init__(self, num_strategy="median", cat_strategy="most_frequent", consider_ordinal_as_cat=False):
        self.consider_ordinal_as_cat = consider_ordinal_as_cat
        assert num_strategy in ("median", "mean")
        fill_value = None
        if cat_strategy != "most_frequent":
            fill_value = cat_strategy
            cat_strategy = "constant"
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.fill_value = fill_value

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X
        self.cols = X_.shape[1]
        self.num_idx, self.cat_idx = parse_cat_col(X_, self.consider_ordinal_as_cat)
        if self.num_idx.size:
            self.num_imputer = SimpleImputer(strategy=self.num_strategy).fit(X_[:, self.num_idx])
        else:
            self.num_imputer = None
        if self.cat_idx.size:
            self.cat_imputer = SimpleImputer(strategy=self.cat_strategy, fill_value=self.fill_value).fit(
                X_[:, self.cat_idx])
        else:
            self.cat_imputer = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X
        assert X_.shape[1] == self.cols
        if self.cat_imputer is not None:
            X_[:, self.cat_idx] = self.cat_imputer.transform(X_[:, self.cat_idx])
        if self.num_imputer is not None:
            X_[:, self.num_idx] = self.num_imputer.transform(X_[:, self.num_idx])
        return X
