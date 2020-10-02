#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from numpy import float64 as npfloat64
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import ParameterGrid
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split
from implicit.evaluation import precision_at_k

"""
Helper functions for Product Recommendation system based on implicit Client
interactions.
"""


def get_prj_path(proj_name):
    """
    Project Absolute Path

    Returns a string describing the absolute path of the project root folder.

    Parameters
    ----------
    proj_name : str
        String describing the project root folder name.
    """
    curr = os.getcwd().split('/')
    path = []
    for folder in curr:
        if folder == proj_name:
            path.append(folder)
            break
        else:
            path.append(folder)
    path = '/'.join(path)
    return path


def get_spr_matrix(df_long, prd_col, cli_col):
    """Calculate Interactions Sparse Matrix

    Given the dataset containing implicit Client-Product interactions in long
    format, returns a sparse representation of it along a transposed version.
    Also transforms cli_col and prd_col in 'category' types for further
    processing.

    Parameters
    ----------
    df_long: pandas DataFrame
        Dataset containing implicit interactions, in long format.
    prd_col: str
        Name of the column containing the ProductIDs.
    cli_col: str
        Name of the column containing the ClientIDs.

    Returns
    -------
    csr_prd_cli_matrix: scipy.csr_matrix
        Sparse CSR representation of df_long, with shape prd_col x cli_col.
    csr_cli_prd_matrix: scipy.csr_matrix
        Sparse CSR representation of df_long, with shape cli_col x prd_col.
    df_cat: pandas DataFrame
        Representation of df_long, with prd_col and cli_col as categories.
    """
    df_cat = df_long.copy()
    if len(df_cat.columns) > 3:
        raise ValueError('The implicit interactions dataset does not have the \
                         right amount of columns. Expected [cli_ids, prd_ids, \
                         implicit_score]')

    score_col = (set(df_cat.columns) - set([prd_col, cli_col])).pop()
    df_cat.dropna(inplace=True)
    df_cat[prd_col] = df_cat[prd_col].astype('category')
    df_cat[cli_col] = df_cat[cli_col].astype('category')

    csr_prd_cli_matrix = coo_matrix((df_cat[score_col],
                                     (df_cat[prd_col].cat.codes.copy(),
                                      df_cat[cli_col].cat.codes.copy())),
                                    dtype=npfloat64).tocsr()
    csr_cli_prd_matrix = coo_matrix((df_cat[score_col],
                                     (df_cat[cli_col].cat.codes.copy(),
                                      df_cat[prd_col].cat.codes.copy())),
                                    dtype=npfloat64).tocsr()

    return csr_prd_cli_matrix, csr_cli_prd_matrix, df_cat


def get_ids(df_cat, prd_col, cli_col):
    """
    Obtain Client and Produtct IDs positions

    Parameters
    ----------
    df_cat: pandas Data Frames
        Representation of df_long, with prd_col and cli_col as categories.
    prd_col: str
        Name of the column containing the ProductIDs.
    cli_col: str
        Name of the column containing the ClientIDs.

    Returns
    -------
    prds: dict
        Dictionary with ProductIDs positions as keys and ProductIDs as values.
    clis: dict
        Dictionary with ClientIDs positions as keys and ClientIDs as values.
    """
    prds = dict(enumerate(df_cat[prd_col].cat.categories))
    clis = dict(enumerate(df_cat[cli_col].cat.categories))

    return prds, clis


def get_prd_prd_recs(prd_ids, model):
    """
    Obtain Recommendations of Similar Products

    Returns a list with Top 10 similar products for each distinct product in
    the dataset. The recommendations are already sorted by affinity with the
    given product.

    Parameters
    ----------
    prd_ids: dict
        Dictionary with ProductIDs positions as keys and ProductIDs as values.
    model: implicit.als.AlternatingLeastSquares model
        Fitted Alternating Leas Squares model from the 'implicit' Python lib.

    Returns
    -------
    prd_recs: pandas DataFrame
    """
    pos_recs = {}
    for pos, prd in prd_ids.items():
        recs = model.similar_items(pos, N=11)
        recs = [prd_ids[el[0]] for el in recs[1:]]
        pos_recs[prd] = recs

    prd_recs = pd.DataFrame(pos_recs.values(),
                            index=pos_recs.keys())
    prd_recs.columns = ['Top' + str(i) + '_rec' for i in range(1, 11)]

    return prd_recs


def get_cli_cli_recs(cli_ids, model):
    """
    Obtain Recommendations of Similar Clients

    Returns a list with Top 10 similar clients for each distinct client in
    the dataset. The recommendations are already sorted by affinity with the
    given client.

    Parameters
    ----------
    cli_ids: dict
        Dictionary with ClientIDs positions as keys and ClientIDs as values.
    model: implicit.als.AlternatingLeastSquares model
        Fitted Alternating Leas Squares model from the 'implicit' Python lib.

    Returns
    -------
    cli_recs: pandas DataFrame
    """
    pos_recs = {}
    for pos, cli in cli_ids.items():
        recs = model.similar_users(pos, N=11)
        recs = [cli_ids[el[0]] for el in recs[1:]]
        pos_recs[cli] = recs

    cli_recs = pd.DataFrame(pos_recs.values(),
                            index=pos_recs.keys())
    cli_recs.columns = ['Top' + str(i) + '_rec' for i in range(1, 11)]

    return cli_recs


def get_prd_cli_recs(csr_cli_prd_matrix, model, cli_ids, prd_ids):
    """
    Obtain Recommendations of Products to Clients

    Returns a list with Top 10 product recommendation to each client in the
    dataset. The recommendations are already sorted by affinity with the
    given client.

    Parameters
    ----------
    csr_cli_prd_matrix: scipy.csr_matrix
        Sparse CSR representation of df_long, with shape cli_col x prd_col.
    model: implicit.als.AlternatingLeastSquares model
        Fitted Alternating Leas Squares model from the 'implicit' Python lib.
    cli_ids: dict
        Dictionary with ClientIDs positions as keys and ClientIDs as values.
    prd_ids: dict
        Dictionary with ProductIDs positions as keys and ProductIDs as values.

    Returns
    -------
    prd_cli_recs: pandas DataFrame
    """
    pos_recs = {}
    for pos, cli in cli_ids.items():
        recs = model.recommend(pos, user_items=csr_cli_prd_matrix, N=11)
        recs = [prd_ids[el[0]] for el in recs[1:]]
        pos_recs[cli] = recs

    prd_cli_recs = pd.DataFrame(pos_recs.values(),
                                index=pos_recs.keys())
    prd_cli_recs.columns = ['Top' + str(i) + '_rec' for i in range(1, 11)]

    return prd_cli_recs


def train_evaluate_als_model(csr_prd_cli_matrix):
    """
    Define, fit and tune ALS model

    Returns an optimized instance of the implicit-ALS model. Implements a Grid
    Search over some hyperparamters. Uses Precision@K as the evaluation metric,
    analyzing 10% of the data given.

    Parameters
    ----------
    csr_prd_cli_matrix: scipy.csr_matrix
        Sparse CSR representation of df_long, with shape prd_col x cli_col.

    Returns
    -------
    model: implicit.als.AlternatingLeastSquares model
    """
    params = {
            'factors': [50, 100, 150],
            'regularization': [0.01, 0.05, 0.1],
            'dtype': [npfloat64],
            'use_native': [True],
            'use_cg': [False],
            'use_gpu': [False],
            'iterations': [15, 30, 50],
            'num_threads': [0],
            'random_state': [42]
            }
    param_grid = ParameterGrid(params)

    df_grid, df_test = train_test_split(csr_prd_cli_matrix,
                                        train_percentage=0.8)
    df_train, df_eval = train_test_split(df_grid, train_percentage=0.8)
    eval_k_size = int(df_eval.shape[0]*0.1)
    test_k_size = int(df_test.shape[0]*0.1)
    grid_score = {}

    for i, grid in enumerate(param_grid):
        m = AlternatingLeastSquares(**grid)
        m.fit(df_train, show_progress=False)
        score = precision_at_k(m, df_train, df_eval, K=eval_k_size,
                               num_threads=0, show_progress=False)
        grid_score[i] = score

    print('Best evaluation Mean Average Precision (@ K={}): {}'
          .format(eval_k_size, pd.Series(grid_score).max()))
    best = pd.Series(grid_score).idxmax()
    best_params = param_grid[best]
    model = AlternatingLeastSquares(**best_params)
    model.fit(csr_prd_cli_matrix)
    test_score = precision_at_k(model, df_train, df_test, K=test_k_size,
                                num_threads=0, show_progress=False)
    print('Best test Mean Average Precision (@ K={}): {}'
          .format(test_k_size, test_score))

    return model
