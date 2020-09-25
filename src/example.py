#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.utils import nonzeros

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)
df = pd.read_csv('data/raw/online_retail.csv')

df.drop(['InvoiceNo', 'Description', 'InvoiceDate', 'Country'],
        axis=1, inplace=True)
df.dropna(inplace=True)
df['total_price'] = df.Quantity * df.UnitPrice
df.drop(['Quantity', 'UnitPrice'], axis=1, inplace=True)

df['StockCode'] = df['StockCode'].astype('category')
df['CustomerID'] = df['CustomerID'].astype('category')

scores = coo_matrix((df.total_price,
                    (df['StockCode'].cat.codes.copy(),
                     df['CustomerID'].cat.codes.copy())))

model = AlternatingLeastSquares(factors=50,
                                regularization=0.01,
                                dtype=np.float64,
                                iterations=50)

model.fit(scores)

prds = dict(enumerate(df['StockCode'].cat.categories))
prds_ids = {r: i for i, r in prds.items()}
