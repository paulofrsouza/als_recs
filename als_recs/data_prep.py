#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd


# Fetching online retail data from UCI repository
def fetch_wrangle_data():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/\
            Online%20Retail.xlsx'
    df = pd.read_excel(url)
    df.drop(['InvoiceNo', 'Description', 'InvoiceDate', 'Country'],
            axis=1, inplace=True)
    df.dropna(inplace=True)
    df['StockCode'] = df.StockCode.astype('object')
    df['CustomerID'] = df.CustomerID.astype('int64')
    df['total_price'] = df.Quantity * df.UnitPrice
    df.drop(['Quantity', 'UnitPrice'], axis=1, inplace=True)
    df.to_csv('../data/online_retail_long.csv', index=False)


if __name__ == '__main__':
    fetch_wrangle_data()
