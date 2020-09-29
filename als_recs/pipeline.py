# -*- coding: utf-8 -*-
# Função principal para recomendações com o ALS-implicit
from als_recs import funcs as als
import pandas as pd
import pickle
import logging


def run_als_rec(df, prd_col, cli_col):
    csr_prd_cli_matrix, csr_cli_prd_matrix, df_cat = als.get_spr_matrix(df,
                                                                        prd_col,
                                                                        cli_col)
    prd_ids, cli_ids = als.get_ids(df_cat, prd_col, cli_col)

    model = als.train_evaluate_als_model(csr_prd_cli_matrix)
    prd_recs = als.get_prd_prd_recs(prd_ids, model)
    cli_recs = als.get_cli_cli_recs(cli_ids, model)
    prd_cli_recs = als.get_prd_cli_recs(csr_cli_prd_matrix,
                                        model, cli_ids, prd_ids)

    # Exportação do modelo
    with open('/path/to/save/models/mdl_als_implicit.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Exportação dos resultados
    prd_recs.to_csv('/path/to/save/prd_prd_als_recs.csv')
    cli_recs.to_csv('/path/to/save/cli_cli_als_recs.csv')
    prd_cli_recs.to_csv('/path/to/save/prd_cli_als_recs.csv')

    return model


if __name__ == '__main__':
    df = pd.read_csv('/path/to/processed/data.csv')
    cli_col = 'cli_col'  # pegar prd_col da CLI
    prd_col = 'prd_col'  # prgar cli_col da CLI
    run_als_rec(df, prd_col, cli_col)
