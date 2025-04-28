import mofax

import pandas as pd


def read_model(model_file, obs):
    m = mofax.mofa_model(model_file)
    
    # there is a bug in the function that loads the metadata which results in the metadata to misaligned
    # so we have to do this manually to make sure the plots are correctly labelled
    m.metadata.drop(
        columns = obs.columns, 
        inplace = True
    )
    m.metadata = m.metadata.merge(
        obs,
        left_index = True,
        right_index = True,
        how = 'inner'
    )
    
    return m
    

def map_human_to_mouse(
    gene_set_table,
    gene_names,
    gene_set_names,
    biomart_mapping
):
    biomart_mapping = biomart_mapping.rename(
        columns = {'external_gene_name': 'symbol'}
    )
    merged = gene_set_table.merge(
        biomart_mapping,
        on = gene_names,
        how = 'inner'
    )
    mapped_gene_names = 'mmusculus_homolog_associated_gene_name'
    mapped_gene_set_table = merged.loc[:, [gene_set_names, mapped_gene_names]]
    mapped_gene_set_table = mapped_gene_set_table.drop_duplicates()
    mapped_gene_set_table.rename(
        columns = {mapped_gene_names: gene_names},
        inplace = True
    )
    return mapped_gene_set_table


def read_gene_sets(
    gene_set_file, 
    gene_set_names, 
    gene_names,
    biomart_mapping = None,
    **kwargs
):
    df = pd.read_csv(
        gene_set_file,
        **kwargs
    )

    if not isinstance(biomart_mapping, type(None)):
        df = map_human_to_mouse(
            df,
            gene_names,
            gene_set_names,
            biomart_mapping
        )

    gene_sets = {}
    for gene_set_name, gene_set_df in df.groupby(gene_set_names):
        gene_sets[gene_set_name] = list(gene_set_df[gene_names].unique())

    return gene_sets
