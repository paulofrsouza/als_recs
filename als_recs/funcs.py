#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from numpy import float64 as npfloat64
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import ParameterGrid

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split
from implicit.evaluation import mean_average_precision_at_k

# Sistema de recomendações de Produtos-para-Produto e de Prdutos-para-Cliente.
# Se baseia na implementação do algoritmo ALS na biblioteca Python `implicit`,
# focada em sistemas de recomendação baseado em scores implícitos. Por 'score
# implícito' se entende qualquer tipo de interação que o cliente teve com o
# produto, onde a interação não se refere a uma avaliação (5 estrelas, p.ex.).

# O algoritmo é extremamente rápido, apresenta implementação em CPython e
# tem amplo suporte para multithreading. Utiliza a representação CSR para
# matrizes esparsas, conforme implementado na lib Scipy, o que também garante
# grande eficiência de memória ao mesmo.

# Montar como um projeto próprio, seguindo o template do cookicutter
# Indicar as necessidades do dataset alimentado: tabela esparsa de scores
# implícitos, com Cliente_ID no index e Produto_ID nas colunas


def get_spr_matrix(df_long, prd_col, cli_col):
    if len(df_long.columns) > 3:
        raise ValueError('The implicit interactions dataset does not have the \
                         right amount of columns. Expected [cli_ids, prd_ids, \
                         implicit_score]')

    score_col = (set(df_long.columns) - set([prd_col, cli_col])).pop()
    df_long.dropna(inplace=True)
    df_long[prd_col] = df_long[prd_col].astype('category')
    df_long[cli_col] = df_long[cli_col].astype('category')

    csr_prd_cli_matrix = coo_matrix((df_long[score_col],
                                     (df_long[prd_col].cat.codes.copy(),
                                      df_long[cli_col].cat.codes.copy())),
                                    dtype=npfloat64).tocsr()
    csr_cli_prd_matrix = coo_matrix((df_long[score_col],
                                     (df_long[cli_col].cat.codes.copy(),
                                      df_long[prd_col].cat.codes.copy())),
                                    dtype=npfloat64).tocsr()

    return csr_prd_cli_matrix, csr_cli_prd_matrix, df_long


def get_ids(df_long, prd_col, cli_col):
    prds = dict(enumerate(df_long[prd_col].cat.categories))
    clis = dict(enumerate(df_long[cli_col].cat.categories))

    return prds, clis


def get_prd_prd_recs(prd_ids, model):
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
    params = {
            'factors': [50, 100, 150],
            'regularization': [0.01, 0.05, 0.1],
            'dtype': [npfloat64],  # prestar atenção nesse parametro
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
        score = mean_average_precision_at_k(m, df_train, df_eval,
                                            K=eval_k_size,
                                            num_threads=0, show_progress=False)
        grid_score[i] = score

    # printar os best param e score para o stdout no futuro
    print('Best evaluation Mean Average Precision (@ K={}): {}'
          .format(eval_k_size, pd.Series(grid_score).max()))
    best = pd.Series(grid_score).idxmax()
    best_params = param_grid[best]
    model = AlternatingLeastSquares(**best_params)
    model.fit(csr_prd_cli_matrix)
    test_score = mean_average_precision_at_k(model,
                                             df_train, df_test,
                                             K=test_k_size,
                                             num_threads=0,
                                             show_progress=False)
    print('Best test Mean Average Precision (@ K={}): {}'
          .format(test_k_size, test_score))

    return model
