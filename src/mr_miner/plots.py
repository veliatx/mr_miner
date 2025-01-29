import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import mr_models
import data_wrappers
from chrom_info import *


def _plot_line_with_error(
    ax, slope: float, se: float, intercept: float = 0, color="k", linestyle="--"
) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if x_max * slope + intercept > y_max:
        # too tall, need to trim along y-axis
        line_x_max = (y_max - intercept) / slope

    else:
        line_x_max = x_max

    if x_min * slope + intercept < y_min:
        # too tall, need to trim along y-axis
        line_x_min = (y_min - intercept) / slope
    else:
        line_x_min = y_min

    center_x = (line_x_min, line_x_max)
    center_y = (line_x_min * slope + intercept, line_x_max * slope + intercept)

    top_y = (
        line_x_min * (slope + se * 2) + intercept,
        line_x_max * (slope + se * 2) + intercept,
    )
    bottom_y = (
        line_x_min * (slope - se * 2) + intercept,
        line_x_max * (slope - se * 2) + intercept,
    )

    ax.plot(center_x, center_y, color=color, label=r"MR $\beta$", linestyle=linestyle)
    ax.fill_between(
        center_x, bottom_y, top_y, alpha=0.5, color=color, label=r"MR $\beta$ 95% CI"
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def plot_model_data(
    mr_data,
    query_tissue,
    query_trait,
    query_gene,
    query_study,
    correlated=False,
    gene_name="",
    snp_color="k",
    mr_color="k",
):
    if correlated:
        model_data, rho = mr_data.get_model_data_correlated(
            query_tissue, query_trait, query_gene
        )[query_study]
        m = mr_models.BurgessPCA(verbose=False)
        slope_param = "beta_IVW_pca"
        se_param = "beta_se_IVW_pca_fixed"
        m.fit(model_data, rho)

    else:
        model_data = mr_data.get_model_data_uncorrelated(
            query_tissue, query_trait, query_gene
        )[query_study]
        m = mr_models.BurgessUncorrelated(verbose=False)
        slope_param = "beta_IVW_uncorrelated"
        se_param = "beta_se_IVW_uncorrelated_random"
        m.fit(model_data)

    print(m.params)

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax.scatter(
        model_data.exposure_beta, model_data.outcome_beta, color=snp_color, label="SNPs"
    )

    first_ci = True
    for spdi, exp_beta, exp_se, out_beta, out_se in iterate_cols(
        model_data, model_data.columns
    ):
        if first_ci:
            ax.plot(
                (exp_beta - exp_se * 2, exp_beta + exp_se * 2),
                (out_beta, out_beta),
                color=snp_color,
                label=r"SNP $\beta$ 95% C.I.",
            )
            first_ci = False
        else:
            ax.plot(
                (exp_beta - exp_se * 2, exp_beta + exp_se * 2),
                (out_beta, out_beta),
                color=snp_color,
            )

        ax.plot(
            (exp_beta, exp_beta),
            (out_beta - out_se * 2, out_beta + out_se * 2),
            color=snp_color,
        )

    square_ax(ax, centered=True)

    _plot_line_with_error(
        ax,
        slope=m.params[slope_param],
        se=m.params[se_param],
        intercept=0,
        color=mr_color,
    )
    if gene_name:
        ax.set_xlabel(f"{gene_name} in {query_tissue}" + r" eQTL $\beta$")
    else:
        ax.set_xlabel(f"{query_gene} in {query_tissue}" + r" eQTL $\beta$")
    ax.set_ylabel(f"{query_trait}" + r" $\beta$")

    ax.legend()

    return fig


def mr_clustermap_per_gene(
    gene_results,
    figsize=(10, 10),
    max_neglogpval=8,
    annot=False,
    include_disease_indication=False,
):
    plot_data = collections.defaultdict(lambda: {})
    for tissue, trait, neg_pval in iterate_cols(
        gene_results, ("tissue", "trait", "neg_log10_pval")
    ):
        # plot_data[trait][tissue] = neg_pval
        plot_data[tissue][trait] = neg_pval

    plot_data = pd.DataFrame(plot_data).fillna(0).sort_index(axis=0).sort_index(axis=1)
    if max_neglogpval is not None:
        plot_data = plot_data.apply(func=lambda x: np.minimum(x, max_neglogpval))

    assert len(gene_results.gene.unique()) == 1
    ensembl_gene = gene_results.gene.unique()[0]
    assert len(gene_results.vtx_id.unique()) == 1
    vtx_id = gene_results.vtx_id.unique()[0]
    assert len(gene_results.gene_name.unique()) == 1
    gene_name = gene_results.gene_name.unique()[0]
    disease = gene_results.disease.unique()[0]

    cgf = sns.clustermap(
        plot_data,
        cmap="YlOrRd",
        robust=True,
        cbar_kws={"label": r"-$\log_{10} p$"},
        vmin=0,
        annot=annot,
        fmt=".1f",
        annot_kws={"size": 10},
        figsize=figsize,
    )
    title = f"{vtx_id} ({gene_name})"
    if include_disease_indication and disease != "":
        title += f"\nImplicated in {disease}"
    cg.fig.suptitle(title)
    # cg.fig.tight_layout()
    return cg


def mr_heatmap_per_gene(
    gene_results,
    output_fpath: Path | str,
    annot: bool = False,
    ax=None,
    max_neglogpval: float = 8,
    include_disease_indication: bool = False,
):
    """
    Generate MR heatmap for a single gene

    Args:
        gene_results: DataFrame with gene results
        output_fpath: Path to save the output plot
        annot: Whether to annotate heatmap cells
        ax: Matplotlib axis to plot on
        max_neglogpval: Maximum -log10 p-value to display
        include_disease_indication: Whether to include disease indication in title
    """
    plot_data = collections.defaultdict(lambda: {})
    for tissue, trait, neg_pval in iterate_cols(
        gene_results, ("tissue", "trait", "neg_log10_pval")
    ):
        plot_data[tissue][trait] = neg_pval

    plot_data = pd.DataFrame(plot_data).fillna(0).sort_index(axis=0).sort_index(axis=1)
    if max_neglogpval is not None:
        plot_data = plot_data.apply(func=lambda x: np.minimum(x, max_neglogpval))

    assert len(gene_results.gene.unique()) == 1
    ensembl_gene = gene_results.gene.unique()[0]
    assert len(gene_results.vtx_id.unique()) == 1
    vtx_id = gene_results.vtx_id.unique()[0]
    assert len(gene_results.gene_name.unique()) == 1
    gene_name = gene_results.gene_name.unique()[0]
    disease = gene_results.disease.unique()[0]

    if annot:
        if ax:
            sns.heatmap(
                plot_data,
                cmap="YlOrRd",
                robust=True,
                cbar_kws={"label": r"-$\log_{10} p$"},
                vmin=0,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 10},
                ax=ax,
            )
        else:
            ax = sns.heatmap(
                plot_data,
                cmap="YlOrRd",
                robust=True,
                cbar_kws={"label": r"-$\log_{10} p$"},
                vmin=0,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 10},
            )

    else:
        if ax:
            sns.heatmap(
                plot_data,
                cmap="YlOrRd",
                robust=True,
                cbar_kws={"label": r"-$\log_{10} p$"},
                vmin=0,
                ax=ax,
            )
        else:
            ax = sns.heatmap(
                plot_data,
                cmap="YlOrRd",
                robust=True,
                cbar_kws={"label": r"-$\log_{10} p$"},
                vmin=0,
            )
    title = f"{vtx_id} ({gene_name})"
    if include_disease_indication and disease != "":
        title += f"\nImplicated in {disease}"
    ax.set_title(title)

    plt.savefig(output_fpath)
    if ax is None:
        plt.close()

    return ax


def plot_manhattan(
    burgess_results,
    chrom_col="chrom",
    pos_col="start",
    y_col="neg_log10_pval",
    y_name=r"$-\log_{10} p$",
    chrom_lengths=CHROM_LENGTHS,
    colors=((0.2, 0.2, 0.5), (0.6, 0.6, 0.9)),
    chrom_padding=0,
    edge_padding=1e7,
    highlights={"chr3": [(80132682, 4.242045, "PLAC9", "r")]},
    point_size=10,
):
    sns.set_style("white")

    fig, ax = plt.subplots(1, figsize=(12, 4))

    pos_offset = 0
    chrom_boundaries = []
    chrom_centers = []
    chrom_labels = []
    total_results_plotted = 0

    for i, chrom in enumerate(chrom_lengths.keys()):
        this_chrom_results = burgess_results.loc[burgess_results[chrom_col] == chrom]
        if this_chrom_results.shape[0] == 0:
            continue

        ax.scatter(
            this_chrom_results[pos_col] + pos_offset,
            this_chrom_results[y_col],
            s=point_size,
            color=colors[i % len(colors)],
        )

        if chrom in highlights:
            for highlight in highlights[chrom]:
                x = highlight[0] + pos_offset
                y = highlight[1]
                ax.scatter(x, y, color=highlight[3], s=point_size)
                ax.annotate(
                    highlight[2], (x, y), color=highlight[3], xytext=(x + 1e6, y + 0.1)
                )

        chrom_boundaries.append(pos_offset - chrom_padding / 2)
        chrom_centers.append(pos_offset + chrom_lengths[chrom] / 2)
        chrom_labels.append(chrom)

        pos_offset += chrom_lengths[chrom] + chrom_padding

    # chrom_boundaries.append(pos_offset)

    ax.set_ylabel(y_name)
    ax.set_xticks(
        chrom_centers,
        labels=chrom_labels,
        rotation=-90,
        ha="center",
        va="center_baseline",
    )

    ylim = [0, ax.get_ylim()[1]]
    # ax.vlines(chrom_boundaries, *ylim, color='darkgrey')
    ax.set_ylim(ylim)
    ax.set_xlim(chrom_boundaries[0] - edge_padding, chrom_boundaries[-1] + edge_padding)

    sns.despine(top=True, right=True)
    fig.tight_layout()

    return fig
