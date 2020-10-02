#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Função principal para recomendações com o ALS-implicit
from als_recs import funcs as als
import pandas as pd
import pickle
import logging


def run_als_rec(input_path, output_path, cli_col, prd_col):
    logging.info('Starting the analysis pipeline.')
    df = pd.read_csv(input_path)
    (csr_prd_cli_matrix,
     csr_cli_prd_matrix,
     df_cat) = als.get_spr_matrix(df, prd_col, cli_col)
    prd_ids, cli_ids = als.get_ids(df_cat, prd_col, cli_col)

    logging.info('Training and evaluating the ALS model.')
    model = als.train_evaluate_als_model(csr_prd_cli_matrix)

    logging.info('Obtaining recommendations.')
    prd_recs = als.get_prd_prd_recs(prd_ids, model)
    cli_recs = als.get_cli_cli_recs(cli_ids, model)
    prd_cli_recs = als.get_prd_cli_recs(csr_cli_prd_matrix,
                                        model, cli_ids, prd_ids)

    logging.info('Exporting trained model and results to {}'
                 .format(output_path))
    if output_path[-1] != '/':
        output_path = ''.join([output_path, '/'])

    with open(output_path + 'mdl_als_implicit.pkl', 'wb') as f:
        pickle.dump(model, f)
    prd_recs.to_csv(output_path + 'prd_prd_als_recs.csv')
    cli_recs.to_csv(output_path + 'cli_cli_als_recs.csv')
    prd_cli_recs.to_csv(output_path + 'prd_cli_als_recs.csv')
    
    return


if __name__ == '__main__':
    df = pd.read_csv('/path/to/processed/data.csv')
    cli_col = 'cli_col'
    prd_col = 'prd_col'
    run_als_rec(df, prd_col, cli_col)
