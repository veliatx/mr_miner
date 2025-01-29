import pandas as pd

import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list


def hierarchical_clustering_optimal_leaf_ordering(df, metric='euclidean', method='centroid', optimal_leaf_order=True):
    """
    Perform hierarchical clustering with optimal leaf ordering on a DataFrame over both axes.

    Parameters:
    df : pandas.DataFrame
        The input data.
    metric : str, optional
        The distance metric to use. Default is 'euclidean'.
    method : str, optional
        The linkage method to use. Default is 'average'.

    Returns:
    pandas.DataFrame
        The reordered DataFrame.
    """
    # Clustering rows
    if df.shape[0] > 1:
        row_dist = pdist(df.values, metric=metric)
        row_linkage = linkage(row_dist, method=method, metric=metric, optimal_ordering=optimal_leaf_order)
        row_order = leaves_list(row_linkage)
    else:
        row_order = [0]

    # Clustering columns
    if df.shape[1] > 1:
        col_dist = pdist(df.values.T, metric=metric)
        col_linkage = linkage(col_dist, method=method, metric=metric, optimal_ordering=optimal_leaf_order)
        col_order = leaves_list(col_linkage)
    else:
        col_order = [0]

    # Reordering the DataFrame
    df_reordered = df.iloc[row_order, col_order]

    # Updating index and columns to reflect new order
    df_reordered.index = df.index[row_order]
    df_reordered.columns = df.columns[col_order]

    return df_reordered


def unmelt(df: pd.DataFrame, row_id_var: str, col_id_var: str, val_var: str, fill_val=0.0) -> pd.DataFrame:
    """
    Convert a melted (long-format) DataFrame to a wide-format matrix.

    Parameters:
    df : pandas.DataFrame
        The input melted DataFrame
    row_id_var : str
        Column name containing row identifiers
    col_id_var : str
        Column name containing column identifiers
    val_var : str
        Column name containing the values
    fill_val : float, optional
        Value to use for missing entries. Default is 0.0

    Returns:
    pandas.DataFrame
        Wide-format DataFrame with row_id_var values as index and col_id_var values as columns
    """
    unmelted_dict = collections.defaultdict(lambda: {})
    for row, col, val in iterate_cols(df, (row_id_var, col_id_var, val_var)):
        unmelted_dict[col][row] = val
    return pd.DataFrame(unmelted_dict).fillna(fill_val)
    

def single_gene_bubble_plot(pruned_results: pd.DataFrame,
                          query_vtx_id: str,
                          output_fpath: Path | str,
                          traits_to_show: list = [],
                          tissues_to_remove: list = [],
                          show_directionality: bool = True,
                          clustering_variable: str = 'mr_beta',
                          optimal_leaf_order: bool = True,
                          show_traits_on_x_axis: bool = False,
                          suptitle_offset: float = 0,
                          hue_variable: str = 'mr_beta',
                          size_variable: str = 'neg_log10_pval'):
    """
    Generate bubble plot for single gene results
    
    Args:
        pruned_results: DataFrame with pruned results
        query_vtx_id: Vertex ID for query
        output_fpath: Path to save the output plot
        traits_to_show: List of traits to include
        tissues_to_remove: List of tissues to exclude
        show_directionality: Whether to show effect directionality
        clustering_variable: Variable to use for clustering
        optimal_leaf_order: Whether to optimize leaf ordering
        show_traits_on_x_axis: Whether to show traits on x-axis
        suptitle_offset: Offset for super title
        hue_variable: Variable to use for point colors
        size_variable: Variable to use for point sizes
    """
    query_gene_name = top_hits_vtx_id_to_name[query_vtx_id]
    
    gene_results = prettify_tissues(get_single_gene_results(pruned_results, query_vtx_id))
    if traits_to_show:
        gene_results = gene_results.loc[my_in1d(gene_results.trait, traits_to_show)]
    if tissues_to_remove:
        gene_results = gene_results.loc[~my_in1d(gene_results.tissue, tissues_to_remove)]

    if min_tissues_per_trait > 1:
        trait_sizes  = gene_results.groupby('trait').size()
        traits_to_keep = trait_sizes.loc[trait_sizes >= min_tissues_per_trait].index
        gene_results = gene_results.loc[my_in1d(gene_results.trait, traits_to_keep)]

    if clustering_variable:
        gene_results = prepare_clustered_gene_results(gene_results, clustering_variable, optimal_leaf_order=optimal_leaf_order)
        
    gene_results['neg_log10_pval'] = np.minimum(gene_results['neg_log10_pval'], max_neglog10_pval)
    gene_results['effect_size'] = np.minimum(gene_results['effect_size'], max_effect_size)
    gene_results['mr_beta'] = np.minimum(gene_results['mr_beta'], max_effect_size)
    gene_results['mr_beta'] = np.maximum(gene_results['mr_beta'], -max_effect_size)
    
    hue_max = np.max(np.abs(gene_results[hue_variable]))
    gene_results = gene_results.rename(columns=column_renamer)
    
    fig, ax = plt.subplots(1, figsize=fig_size)

    x_var = column_renamer['tissue']
    y_var = column_renamer['trait']
    if show_traits_on_x_axis:
        x_var, y_var = y_var, x_var

    if show_directionality:
        markers = {-1:'v', 1:'^'}
        style = 'direction'
    else:
        markers = True
        style = None
        
    sns.scatterplot(gene_results, 
                    x=x_var,
                    y=y_var,
                    ax=ax,
                    style=style, 
                    hue=column_renamer[hue_variable],
                    hue_norm=(-hue_max, hue_max),
                    markers=markers,
                    size=column_renamer[size_variable],
                    sizes=(min_size, max_size), palette=pal, 
                    legend=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
    fig.suptitle(f'{query_vtx_id} ({query_gene_name})\nMendelian randomization results', y=0.98+suptitle_offset)
    
    if show_watermark:
        ax.text(0.5, 0.5, style_name, transform=ax.transAxes,
                fontsize=40, color='gray', alpha=0.5,
                ha='center', va='center', rotation=30)
    
    fig.tight_layout()
    fig.savefig(output_fpath, dpi=300, bbox_inches='tight')
    plt.close()
    

def prepare_clustered_gene_results(gene_results: pd.DataFrame, clustering_variable: str='mr_beta', optimal_leaf_order: bool=True) -> pd.DataFrame:
    """
    Prepare gene results for visualization by performing hierarchical clustering.

    Parameters:
    gene_results : pandas.DataFrame
        DataFrame containing gene analysis results
    clustering_variable : str, optional
        Variable to use for clustering. Default is 'mr_beta'
    optimal_leaf_order : bool, optional
        Whether to perform optimal leaf ordering. Default is True

    Returns:
    pandas.DataFrame
        DataFrame with categorical tissue and trait columns ordered by clustering
    """
    clustered_df = hierarchical_clustering_optimal_leaf_ordering(unmelt(gene_results, 'trait', 'tissue', clustering_variable), optimal_leaf_order=optimal_leaf_order)
    clustered_df.index.name = 'trait'
    clustered_df.columns.name = 'tissue'
    clustered_trait_order = clustered_df.index
    clustered_tissue_order = clustered_df.columns
    
    gene_results['tissue'] = pd.Categorical(
        gene_results['tissue'],
        categories=clustered_tissue_order,
        ordered=True
    )
    
    gene_results['trait'] = pd.Categorical(
        gene_results['trait'],
        categories=clustered_trait_order,
        ordered=True
    )
    return gene_results    


traits_to_show = {'VTX-0661740':[#'Highest math class taken (MTAG) [MTAG]',
                                 'Age-related macular degeneration',
                                 # 'Arm fat percentage (left)',
                                 'Arm fat percentage (right)',
                                 'Body fat percentage',
                                 'Diastolic blood pressure',
                                 'Hemoglobin A1c levels',
                                 #'High light scatter reticulocyte count',
                                 #'Hip circumference',
                                 # 'Hip circumference [EA]',
                                 'Immature fraction of reticulocytes',
                                 'Impedance of arm (right)',
                                 'Lymphocyte counts',
                                 #'Sex hormone-binding globulin levels',
                                 #'Sex hormone-binding globulin levels adjusted for BMI',
                                 'Systolic blood pressure [EA]',
                                 'Total cholesterol levels',
                                 # 'Trunk fat mass',
                                 'Trunk fat percentage',
                                 'Type 2 diabetes',
                                 #'Type 2 diabetes (adjusted for BMI)',
                                 # 'Type 2 diabetes [EA]',
                                 # 'Waist circumference adjusted for BMI (adjusted for smoking behaviour) [women]',
                                 # 'Waist circumference adjusted for BMI in active individuals [EA,women]',
                                 # 'Waist circumference adjusted for BMI in non-smokers [women]',
                                 # 'Waist circumference adjusted for body mass index [EA, women]',
                                 # 'Waist circumference adjusted for body mass index [EA,women]',
                                 # 'Waist circumference adjusted for body mass index [EA]',
                                 #'Waist-hip ratio',
                                 # 'Waist-hip ratio [EA, women]',
                                 # 'Waist-hip ratio [EA]',
                                 'Waist-to-hip ratio adjusted for BMI',
                                 # 'Waist-to-hip ratio adjusted for body mass index',
                                 # 'Waist-to-hip ratio adjusted for body mass index [AA, Women]',
                                 # 'Waist-to-hip ratio adjusted for body mass index [AA]',
                                 # 'Waist-to-hip ratio adjusted for body mass index [EA]',
                                 'White blood cell count',
                                 'Whole body fat mass',
                                 'Leg fat percentage (right)',
                                 'Liver enzyme levels (alanine transaminase)',
                                 # 'FEV1',
                                 'Forced expiratory volume in 1-second (fev1), best measure'
                                ],
                  'VTX-0017706':[
                         # '3mm strong meridian (left)',
                         # '3mm strong meridian (right)',
                         # '6mm strong meridian (left)',
                         # '6mm strong meridian (right)',
                         'Alanine aminotransferase levels',
                         'Apolipoprotein A1 levels',
                         #'Apolipoprotein B levels',
                         'Appendicular lean mass',
                         # 'Arm fat mass (left)',
                         'Arm fat mass (right)',
                         #'Ascending aorta maximum area',
                         'Asthma',
                         'Atopic dermatitis [EA, fixed effects]',
                         'Basophil count',
                         # 'Basophil percentage of white cells',
                         'Bipolar disorder (MTAG)',
                         # 'Blood protein levels [EA, Extracellular matrix protein 1]',
                         #'Blood protein levels [ECM1, 3366_51_2]',
                         # 'Blood protein levels [Extracellular matrix protein 1]',
                         #'Blood protein levels [FN1, 3434_34_1]',
                         # 'Blood protein levels [FN1, 3435_53_2]',
                         #'Blood protein levels [SERPINF2, 3024_18_2]',
                         'C-reactive protein levels',
                         'Chronotype',
                         'Creatinine levels',
                         'Direct bilirubin levels',
                         # 'Direct low density lipoprotein cholesterol levels',
                         'Estimated glomerular filtration rate',
                         # 'Estimated glomerular filtration rate in non-diabetics',
                         #'FEV1',
                         'Granulocyte percentage of myeloid white cells',
                         'HDL cholesterol',
                         # 'HDL cholesterol levels',
                         # 'Hand grip strength (left)',
                         #'Hand grip strength (right)',
                         #'Headache | pain type(s) experienced in last month',
                         'Heel bone mineral density',
                         # 'Heel bone mineral density T score',
                         #'Height',
                         'Hematocrit',
                         #'Highest math class taken (MTAG) [MTAG]',
                         'Hip pain | pain type(s) experienced in last month',
                         # 'Impedance of arm (right)',
                         # 'Impedance of leg (left)',
                         # 'Impedance of leg (right)',
                         'Impedance of whole body',
                         'LDL cholesterol levels',
                         # 'Leg fat mass (left)',
                         'Leg fat mass (right)',
                         # 'Liver enzyme levels (alanine transaminase)',
                         # 'Low density lipoprotein cholesterol levels',
                         #'Lung function (FVC)',
                         #'Lymphocyte percentage',
                         # 'Lymphocyte percentage of white cells',
                         # 'Mean corpuscular hemoglobin',
                         # 'Mean corpuscular volume',
                         # 'Mean platelet volume',
                         #'Migraine',
                         'Monocyte count',
                         # 'Monocyte percentage of white cells',
                         # 'Morning person',
                         # 'Morning vs. evening chronotype',
                         # 'Morning/evening person (chronotype)',
                         # 'Morningness',
                         'Multisite chronic pain',
                         'Neutrophil count',
                         # 'None of the above | medication for pain relief, constipation, heartburn',
                         'Osteoarthritis',
                         # 'Osteoarthritis (hip)',
                         # 'Osteoarthrosis',
                         'Peak expiratory flow',
                         # 'Peak expiratory flow (pef)',
                         # 'Pediatric bone mineral content (hip) [Males]',
                         # 'Place of birth in uk - east co-ordinate',
                         'Platelet count',
                         # 'Platelet crit',
                         # 'Plateletcrit',
                         # 'Protein quantitative trait loci (liver) [COL18A1]',
                         # 'Red cell distribution width',
                         # 'Schizophrenia',
                         'Schizophrenia (MTAG)',
                         # 'Self-reported math ability',
                         #'Self-reported math ability (MTAG) [MTAG]',
                         'Serum 25-Hydroxyvitamin D levels',
                         # 'Serum levels of protein ECM1',
                         # 'Serum levels of protein FN1',
                         # 'Serum levels of protein TNFRSF13B',
                         'Serum urea levels',
                         'Sex hormone-binding globulin levels',
                         # 'Sex hormone-binding globulin levels adjusted for BMI',
                         #Spontaneous coronary artery dissection',
                         'Total testosterone levels',
                         'Triglyceride levels',
                         'Type 2 diabetes',
                         'Urate levels',
                         #'Use both right and left hands equally | handedness (chirality/laterality)',
                         #'Vertex-wise cortical surface area',
                         #'Vertex-wise cortical thickness',
                         'Waist-to-hip ratio adjusted for BMI',
                         'White blood cell count']                     
                 }

style_kwargs = {'Style 1': {'hue_variable':'neg_log10_pval',
                            'size_variable':'effect_size',
                            'pal':'viridis_r'
                        },
                'Style 2': {'hue_variable':'mr_beta',
                            'size_variable':'neg_log10_pval',
                            'pal':'coolwarm'
                        }                
                }


style_name = 'Style 1'


single_gene_bubble_plot(pruned_results, 
                        query_vtx_id=query_vtx_id,
                        output_fpath=paths.pipeline.figures.basepath.joinpath(f'{query_vtx_id}_curated_traits_bubble_{style_name.replace(' ', '_')}.png'),                       
                        traits_to_show=traits_to_show[query_vtx_id],
                        fig_size=(2, 4.5),
                        perform_clustering=True,
                        clustering_variable='mr_beta',
                        optimal_leaf_order=True,
                        max_effect_size=1,
                         **style_kwargs[style_name],
                        )