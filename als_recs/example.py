#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
from als_recs import funcs as als
from implicit.als import AlternatingLeastSquares
from IPython.display import display
from numpy import float64 as npfloat64

import warnings
warnings.filterwarnings('ignore')


def als_recs_example():
    # Fetching online retail data from UCI repository
    # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
    # df = pd.read_excel(url)
    df = pd.read_csv('../data/raw/online_retail.csv')
    prd_col = 'StockCode'
    cli_col = 'CustomerID'

    df.drop(['InvoiceNo', 'Description', 'InvoiceDate', 'Country'],
            axis=1, inplace=True)
    df.dropna(inplace=True)
    df['CustomerID'] = df.CustomerID.astype('int64')
    df['total_price'] = df.Quantity * df.UnitPrice
    df.drop(['Quantity', 'UnitPrice'], axis=1, inplace=True)

    (csr_prd_cli_matrix,
     csr_cli_prd_matrix,
     df_cat) = als.get_spr_matrix(df, prd_col, cli_col)
    prd_ids, cli_ids = als.get_ids(df_cat, prd_col, cli_col)

    model = als.train_evaluate_als_model(csr_prd_cli_matrix)
    """
    model = AlternatingLeastSquares(factors=50,
                                    regularization=0.01,
                                    dtype=npfloat64,
                                    iterations=50,
                                    use_native=True,
                                    num_threads=0,
                                    use_gpu=False)
    model.fit(csr_prd_cli_matrix)
    """

    prd_recs = als.get_prd_prd_recs(prd_ids, model)
    cli_recs = als.get_cli_cli_recs(cli_ids, model)
    prd_cli_recs = als.get_prd_cli_recs(csr_cli_prd_matrix,
                                        model, cli_ids, prd_ids)

    print('\n' + '-'*30)
    print("Similar Products recommendations\n")
    display(prd_recs.head())
    print('\n\n' + '-'*30)
    print("Similar Clients recommendations\n")
    display(cli_recs.head())
    print('\n\n' + '-'*30)
    print("Recommendations of Products to each Client\n")
    display(prd_cli_recs.head())


if __name__ == '__main__':
    als_recs_example()
