#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import logging
import click
from als_recs import pipeline, example

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)


def run_tutorial(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    example.als_recs_example()
    ctx.exit()


@click.command()
@click.option('--tutorial', is_flag=True, callback=run_tutorial,
              expose_value=False, is_eager=True,
              help="tutorial mode runs a recommendation example\
                   using the infamous 'Online Retail' dataset from UCI's \
                   repositories.")
@click.option('--input_path',
              type=click.Path(exists=True, readable=True),
              prompt='Absolute path to input file',
              help='Absolute path to file containing the input csv file\
                    with implicit scores. File Must be readable.')
@click.option('--output_path',
              type=click.Path(exists=True, writable=True),
              prompt='Absolute path to output folder',
              help='Absolute path to folder to contain output csv files\
              and pickled als model. Folder must be writable.')
@click.option('--cli_col',
              type=click.STRING,
              prompt='Client IDs column',
              help='Column in the input dataset containing the Clients IDs')
@click.option('--prd_col',
              type=click.STRING,
              prompt='Product IDs column',
              help='Column in the input dataset containing the Products IDs')
def als_recommendation(input_path, output_path, cli_col, prd_col):
    """Product recommendation system based on client's implicit interactions
    """
    pipeline.run_als_rec(input_path, output_path, cli_col, prd_col)


if __name__ == '__main__':
    als_recommendation()
