import pandas as pd
import numpy as np
import reComBat as rc
import statsmodels.api as sm

from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection


def filter_high_nan_features_by_group(
    df, 
    metadata, 
    grouping_columns, 
    allowed_nan_fraction = 0.5
):
    merged = df.merge(
        metadata[grouping_columns],
        left_index = True,
        right_index = True,
        how = 'inner'
    )
    value_counts = merged.groupby(grouping_columns).count()
    value_fractions = value_counts.div(
        value_counts.max(axis = 1), 
        axis = 0
    )
    return df.loc[:, (value_fractions >= (1 - allowed_nan_fraction)).all(axis = 0)]


def fillna(group, group_means, group_stds):
    for i, row in group.iterrows():
        row_nan_features = row.isna()
        fill_values = np.random.normal(
            loc = group_means[row_nan_features],
            scale = group_stds[row_nan_features]
        )
        fill_values[fill_values < 0] = 0
        group.loc[i, row_nan_features] = fill_values
    
    return group


def impute_group_nans(df, metadata, grouping_columns):
    merged = df.merge(
        metadata[grouping_columns],
        left_index = True,
        right_index = True,
        how = 'inner'
    )
    # group_feature_medians = merged.groupby(grouping_columns).median()
    group_feature_means = merged.groupby(grouping_columns).mean()
    group_feature_stds = merged.groupby(grouping_columns).std()
    imputed_df = pd.concat(
        [
            fillna(
                group_df,
                group_feature_means.loc[group_index, :],
                group_feature_stds.loc[group_index, :]
            )
            for group_index, group_df 
            in merged.groupby(grouping_columns)
        ]
    )
    return imputed_df.reindex(df.index).drop(columns = grouping_columns)


def remove_batch_effect(data, metadata, wanted_variation_columns, batch_column):
    log_data = np.log1p(data)
    standardized_data = ((log_data.T - log_data.T.mean()) / log_data.T.std()).T

    model = rc.reComBat(
        parametric=True,
        model='ridge',
        config={'alpha':1e-9},
        conv_criterion=1e-4,
        max_iter=1000,
        n_jobs=1,
        mean_only=False,         
        optimize_params=True,
        reference_batch=None,
        verbose=True                
    )
    batches = metadata.loc[:, batch_column]
    wanted_variation_design = metadata.loc[:, wanted_variation_columns]
    X = model.fit_transform(
        standardized_data,
        batches,
        X = wanted_variation_design
    )
    return X


def quantile_normalize_data(data):
    ranked_data = data.rank(axis = 1)
    sorted_data = pd.concat(
        [row.sort_values(ignore_index = True) for _, row in data.iterrows()],
        axis = 1
    )
    rank_values = sorted_data.mean(axis = 1)
    quantile_normalized_data = ranked_data.copy()
    for rank, value in rank_values.items():
        quantile_normalized_data.replace(rank, value, inplace = True)
    
    return quantile_normalized_data


def two_way_anova(data, feature, cov1, cov2):
    lm = ols(
        f'{feature} ~ C({cov1})*C({cov2})',
        data = data
    )
    model = lm.fit()
    return sm.stats.anova_lm(model, typ = 2)


def data_long_format(
    data, 
    feature, 
    metadata, 
    covariate_columns,
):
    annotated_data = data.loc[:, [feature]].merge(
        metadata.loc[:, covariate_columns],
        left_index = True,
        right_index = True,
        how = 'inner'
    )
    
    annotated_data.rename(
        columns = {feature: replace_any_patsy_operator(feature)},
        inplace = True
    )
        
    return annotated_data


def replace_any_patsy_operator(string):
    # this is necessary due to the way patsy parses formulas
    # ';' is not really an operator but also results in an error so we replace it
    operators = ['-', '+', '/', ':', '~' '**', ';']
    for operator in operators:
        string = string.replace(operator, '')
    
    return string


def featurewise_two_way_anova(
    data,
    metadata,
    covariate_columns,
    allowed_nan_fraction = 1
):
    # rename features to make sure there is no clash with anything
    renaming = {feature: f'feature{i}' for i, feature in enumerate(data.columns)}
    data = data.rename(columns = renaming)
    filtered_data = filter_high_nan_features_by_group(
        data,
        metadata,
        covariate_columns,
        allowed_nan_fraction
    )
    
    anova_pvalues = dict()
    cov1, cov2 = covariate_columns
    for feature in filtered_data.columns:
        anova_result = two_way_anova(
            data_long_format(
                filtered_data, 
                feature, 
                metadata, 
                covariate_columns
            ),
            replace_any_patsy_operator(feature),
            cov1, cov2
        )
        anova_pvalues[feature] = anova_result.loc[:, 'PR(>F)']
    
    anova_pvalues = pd.DataFrame.from_dict(
        anova_pvalues,
        orient = 'index'
    )

    anova_pvalues.drop(
        columns = ['Residual'],
        inplace = True
    )

    anova_pvalues.rename(
        columns = {
            column: 'pval_' + column for column in anova_pvalues.columns
        },
        inplace = True
    )

    for pval_column in anova_pvalues.columns:
        _, padj = fdrcorrection(
            anova_pvalues[pval_column]
        )
        padj_column = pval_column.replace('pval', 'padj')
        anova_pvalues[padj_column] = padj
    
    return anova_pvalues.rename(index = {v: k for k, v in renaming.items()})
    
