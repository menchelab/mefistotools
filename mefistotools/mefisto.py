import anndata as ad
import muon as mu
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from . import plot, preprocess


def preprocess_and_fit_mefisto(
    data, 
    metadata, 
    preprocess_kwargs,
    adata_save_file, 
    plot_kwargs,
    recombat_kwargs,
    mefisto_kwargs,
    maxiter = 1000,
    do_combat = False
):
    print('filtering')
    filtered_df = preprocess.filter_high_nan_features_by_group(
        data,
        metadata,
        preprocess_kwargs['grouping_columns'],
        preprocess_kwargs['allowed_nan_fraction']
    )
    
    print('imputing')
    imputed_df = preprocess.impute_group_nans(
        filtered_df,
        metadata,
        preprocess_kwargs['grouping_columns']
    )
    
    if do_combat:
        print('correcting batch')
        batch_corrected_df = preprocess.remove_batch_effect(
            imputed_df, 
            metadata, 
            recombat_kwargs['batch_wanted_variation_columns'],
            recombat_kwargs['batch_column']
        )
    
    print('quantile normalize')
    qn_df = preprocess.quantile_normalize_data(imputed_df)
    
    adata = ad.AnnData(
        X = filtered_df,
        var = pd.DataFrame(index = filtered_df.columns),
        obs = metadata.reindex(filtered_df.index)
    )
    adata.layers['raw'] = adata.X.copy()
    adata.layers['imputed'] = imputed_df.copy()
    
    if do_combat:
        adata.layers['batch_corrected'] = batch_corrected_df.copy()
        
    adata.layers['quantile_normalized'] = qn_df.copy()
    
    adata.write(adata_save_file)
    
    print('plotting reduced dimensions')
    for layer in adata.layers:
        figs = plot.plot_reduced_dimensions(
            adata,
            layer = layer,
            n_cols = plot_kwargs['n_cols'],
            n_rows = plot_kwargs['n_rows']
        )
        
        for fig, dim_red_type in zip(figs, ['pca', 'umap']):
            fig.savefig(
                plot_kwargs['plot_save_file'].format(
                    dim_red_type,
                    layer
                )
            )
            plt.close(fig)
    
    # is needed since mefisto can only interpret numbers for smooth covariate
    factors = {
        k: i for i, k in enumerate(mefisto_kwargs['time_ordering'])
    }
    adata.obs['timefactor'] = adata.obs[mefisto_kwargs['time_column']].apply(
        lambda x: factors[x]
    )
    
    adata.X = adata.layers[mefisto_kwargs['layer']].copy()
    sc.pp.log1p(adata)
    
    mu.tl.mofa(
        adata, 
        n_factors = mefisto_kwargs['n_factors'], # number of factors to fit
        groups_label = mefisto_kwargs['groups_label'], # column of adata.obs to use for data grouping
        center_groups = False,
        n_iterations = maxiter,
        smooth_covariate = mefisto_kwargs['smooth_covariate'], # column to use as time variable
        smooth_kwargs = mefisto_kwargs['smooth_kwargs'], # additional arguments for MEFISTO
        outfile = mefisto_kwargs['mefisto_save_file'],
        seed = 2023
    )
