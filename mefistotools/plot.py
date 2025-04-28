import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import itertools as it
import pandas as pd

import mofax
import gc

from . import io
from scipy.stats import zscore

mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['pdf.fonttype'] = 42


def plot(adata, plot_func, n_rows, n_cols, **kwargs):
    fig, axs = plt.subplots(n_rows, n_cols)

    for ax, column in zip(axs.reshape(n_rows * n_cols), adata.obs.columns):
        plot_func(
            adata,
            color = column,
            ax = ax,
            show = False,
            **kwargs
        )
        
    fig.set_figwidth(10 * n_cols)
    fig.set_figheight(10 * n_rows)
    fig.tight_layout()
    
    return fig
    
    
def plot_reduced_dimensions(
    adata, 
    n_rows, 
    n_cols, 
    layer = None, 
    use_highly_variable = False, 
    n_hvg = 4000
):
    if layer:
        adata.X = adata.layers[layer].copy()
    
    sc.pp.log1p(adata)
    
    if np.isnan(adata.X).any():
        adata.X = np.nan_to_num(adata.X)
    
    if use_highly_variable:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes = n_hvg
        )
        
    sc.pp.pca(
        adata, 
        n_comps = 40, 
        svd_solver = 'arpack',
        use_highly_variable = use_highly_variable
    )

    fig_pca = plot(adata, sc.pl.pca, n_rows, n_cols)

    sc.pp.neighbors(
        adata,
        use_rep = 'X_pca'
    )
    sc.tl.umap(adata)

    fig_umap = plot(adata, sc.pl.umap, n_rows, n_cols)
    return fig_pca, fig_umap


def plot_factors(m, x, y, color, n_rows, n_cols, alpha = 1):
    data = m.fetch_values([x, color, *y])
    
    fig, axs = plt.subplots(n_rows, n_cols)
    for ax, factor in zip(
        axs.reshape(n_rows * n_cols), 
        data.columns[data.columns.str.startswith('Factor')]
    ):
        sns.scatterplot(
            data = data,
            x = x,
            y = factor,
            hue = color,
            ax = ax,
            palette = 'husl'
        )

        sns.lineplot(
            data = data,
            x = x,
            y = factor, 
            hue = color,
            ax = ax,
            estimator = 'mean',
            palette = 'husl'
        )
        
        ax.set_title(factor)
    
    fig.set_figwidth(5.5 * n_cols)
    fig.set_figheight(5 * n_rows)
    fig.tight_layout()
    
    return fig


def plot_annotated_factor_combination(factors_and_metadata, f1, f2, axs):
    metadata_columns_idx = ~factors_and_metadata.columns.str.startswith('Factor')
    metadata_columns = factors_and_metadata.columns[metadata_columns_idx]
    for ax, metadata_column in zip(axs, metadata_columns):
        sns.scatterplot(
            data = factors_and_metadata,
            x = f1,
            y = f2,
            hue = metadata_column,
            ax = ax
        )
        ax.set_title(metadata_column)
        unique_values = factors_and_metadata[metadata_column].unique()
        value_lengths = [len(val) for val in unique_values if isinstance(val, str)]
        max_value_length = np.max(value_lengths) if value_lengths else 0
        if len(unique_values) > 5 or max_value_length > 15:
            ax.legend().remove()


def plot_model_evaluations(
    model_file, 
    obs, 
    plot_prefix, 
    n_factors = 10, 
    n_rows_factors = 2, 
    n_cols_factors = 5,
    group_column = 'group',
    groups = True
):
    m = io.read_model(model_file, obs)
    
    """
    The smoothness of a factor indicates the amount of non-temporal variation captured by each factor. 
    Values near 1 imply high smoothness and indicate that the factor captures temproal variation while 
    values near 0 indicate the factor captures a lot of noise of non-temporal variation. In this case, 
    all learned factors seem to capture a lot of temporal variation.
    """
    ax = mofax.plot_smoothness(m)
    ax.set_title('Factor smoothness')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(plot_prefix + '_factor_smoothness.pdf')
    plt.close(fig)
    
    if groups:
        """
        The group kernel plots show the correlation of the temporal patterns between the groups captured by a given factor.
        So the plot tells us how much a given temporal pattern captured by a factor is shared between groups.
        """
        ax = mofax.plot_sharedness(m)
        ax.set_title('Factor sharedness')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(plot_prefix + '_factor_sharedness.pdf')
        plt.close(fig)
        
        """
        Same as above but mapped to a value between 0 and 1 indicating how much the samples share the same pattern captured by each factor.
        This is a pairwise comparison. While the above plot seems to be the mean of the pairwise sharedness
        """
        ax = mofax.plot_group_kernel(m)
        fig = ax.get_figure()
        fig.savefig(plot_prefix + '_group_kernel_plots.pdf')
        plt.close(fig)
        
    fig = mofax.plot_r2(m, vmax = 10)
    fig.savefig(plot_prefix + '_explained_variance.pdf')
    plt.close(fig)
    
    factors = list(range(n_factors)) if isinstance(n_factors, int) else n_factors
    fig = plot_factors(
        m, 
        'timefactor', 
        factors, 
        group_column, 
        n_rows_factors, 
        n_cols_factors, 
        0.5
    )
    fig.savefig(plot_prefix + '_factors.pdf')
    plt.close(fig)
    
    factors_and_metadata = m.fetch_values(
        [*factors] + m.metadata.columns[:-1].to_list()
    )
    for f1, f2 in it.combinations(factors, 2):
        # this is very dataspecific and needs to be changed
        # but for time reasons I leave it hardcoded for now
        fig, axs = plt.subplots(4, 4)
        plot_annotated_factor_combination(
            factors_and_metadata, 
            f'Factor{f1+1}',
            f'Factor{f2+1}',
            axs.reshape(16)
        )
        fig.set_figwidth(5.5 * 4)
        fig.set_figheight(5 * 4)
        fig.tight_layout()
        fig.savefig(plot_prefix + f'_factor{f1+1}_factor{f2+1}_annotated.pdf')
        plt.close(fig)
        gc.collect()
    
    m.close()
    
    del factors, m
    gc.collect()


def expand_gene_names(factor_weights):
    expanded_weights = []
    for gene_name, weights in factor_weights.iterrows():
        split_name = gene_name.split(';')
        weight_list = weights.to_list()
        for name in split_name:
            expanded_weights.append(
                [name, *weight_list]
            )

    expanded_weights = pd.DataFrame(
        expanded_weights,
        columns = ['gene_name'] + factor_weights.columns.to_list()
    )
    return expanded_weights.set_index('gene_name')


def plot_factor_values(model):
    factor_weights = expand_gene_names(
        model.get_weights(df = True)
    )
    factor_weights_zscore = factor_weights.apply(zscore, axis = 0)

    fig, axs = plt.subplots(2, len(factor_weights.columns))
    for axs_row, weight_df in zip(
        axs, 
        [factor_weights, factor_weights_zscore]
    ):
        for ax, column in zip(axs_row, weight_df.columns):
            sns.histplot(
                data = weight_df,
                x = column,
                ax = ax,
                bins = 25
            )
    
    fig.set_figheight(5)
    fig.set_figwidth(2.5 * len(factor_weights.columns))
    fig.tight_layout()

    return fig