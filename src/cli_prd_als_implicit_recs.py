#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from numpy import float64 as npfloat64
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares

import pickle

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


def get_ids(df_long, prd_col, cli_col):
    prds = dict(enumerate(df_long[prd_col].cat.categories))
    clis = dict(enumerate(df_long[cli_col].cat.categories))

    return prds, clis


def get_spr_matrix(df_long, prd_col, cli_col):
    if len(df_long.columns) > 3:
        raise ValueError('The implicit interactions dataset does not have the \
                         right amount of columns. Expected [cli_ids, prd_ids, \
                         implicit_score]')

    score_col = (set(df_long.columns) - set([prd_col, cli_col])).pop()
    df_long.dropna(inplace=True)
    df_long[prd_col] = df_long[prd_col].astype('category')
    df_long[cli_col] = df_long[cli_col].astype('category')

    coo_prd_cli_matrix = coo_matrix((df_long[score_col],
                                     (df_long[prd_col].cat.codes.copy(),
                                      df_long[cli_col].cat.codes.copy())))
    coo_cli_prd_matrix = coo_matrix((df_long.socre_col,
                                     (df_long[cli_col].cat.codes.copy(),
                                      df_long[prd_col].cat.codes.copy())))

    return coo_prd_cli_matrix, coo_cli_prd_matrix, df_long


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


# Função para extração de recomendações de prouto-para-produto. Inclui
# construção do DataFrame de entrega
def get_prd_for_prd_recs(prd_look, model):
    pos_recs = prd_look.index \
                        .map(lambda x: model.similar_items(x, N=11)) \
                        .tolist()
    # removendo o prd da própria recomendação
    pos_recs = [el[1:] for el in pos_recs]
    pos_recs_dict = dict()

    for pos in range(prd_look.index.stop):
        pos_recs_dict[prd_look.iloc[pos]] = [prd_look.iloc[el[0]]
                                             for el in pos_recs[pos]]

    prd_recs = pd.DataFrame(pos_recs_dict.values(), index=pos_recs_dict.keys())
    prd_recs.columns = ['Top' + str(i) + '_rec' for i in range(1, 11)]

    return prd_recs


# Função para extração de recomendações de prouto-para-cliente. Inclui
# construção do DataFrame de entrega
def get_prd_for_cli_recs(cli_look, prd_look, model, user_items_df):
    pos_recs = cli_look.index \
                    .map(lambda x: model.recommend(x, user_items=user_items_df,
                                                   N=10)) \
                    .tolist()

    pos_recs_dict = dict()
    for pos in range(cli_look.index.stop):
        pos_recs_dict[cli_look.iloc[pos]] = [prd_look.iloc[el[0]]
                                             for el in pos_recs[pos]]

    prd_recs = pd.DataFrame(pos_recs_dict.values(), index=pos_recs_dict.keys())
    prd_recs.columns = ['Top' + str(i) + '_rec' for i in range(1, 11)]

    return prd_recs


# Função principal para recomendações com o ALS-implicit
def run_prd_als_rec(df):
    # Criando referência de Produto_ID e Cliente_ID com o valor posiocional de
    # cada um. Necessário para consulta no algoritmo, que trabalha com os
    # posicionais apenas
    cli_look = pd.Series(df.index.tolist(), name='cli_id')
    prd_look = pd.Series([el[1] for el in df.columns.tolist()], name='prd_id')

    # Criando representação em CSR da matriz esparsa de scores para treinamento
    # do modelo. A mesma precisa estar no formato Produtos X Clientes, logo a
    # aplicação de Transposta a mesma.
    spr_prd_cli_df = csr_matrix(df.T.values)

    # Para recomendação de produtos a clientes, é necessário que a matriz
    # esparsa de scores esteja no formato Clientes x Produtos
    spr_cli_prd_df = csr_matrix(df.values)

    # Forçando que o modelo não use aceleração de GPU, visando evitar problemas
    # de configuração em diferentes ambientes. Utilizando máximo número de
    # threads
    model = AlternatingLeastSquares(factors=50,
                                    regularization=0.01,
                                    dtype=npfloat64,
                                    iterations=50,
                                    use_native=True,
                                    num_threads=0,
                                    use_gpu=False)

    # Fittando o modelo
    model.fit(spr_prd_cli_df)

    # Obtendo o DataFrame de recomendação de Produtos para Produtos
    prd_recs = get_prd_for_prd_recs(prd_look, model)

    # Obtendo o DataFrame de recomendação de Produtos para Clientes
    cli_recs = get_prd_for_cli_recs(cli_look, prd_look, model, spr_cli_prd_df)

    # Exportação do modelo
    with open('/path/to/save/models/mdl_als_implicit.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Exportação dos resultados
    prd_recs.to_csv('/path/to/save/prd_prd_als_recs.csv')
    cli_recs.to_csv('/path/to/save/prd_cli_als_recs.csv')

    return model


if __name__ == '__main__':
    df = pd.read_csv('/path/to/processed/data.csv')
    run_prd_als_rec(df)
