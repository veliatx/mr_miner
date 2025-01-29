import logging
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from mr_miner import mr_models
from mr_miner.constants import MIN_PVALUE
from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection

logger = logging.getLogger(__name__)


# def pick_best_betas(
#     result_df,
#     beta_result_column_ordering=(
#         "beta_IVW_pca",
#         "beta_IVW_cholesky",
#         "beta_IVW_uncorrelated",
#     ),
# ):
#     best_betas = np.full(shape=result_df.shape[0], fill_value=np.nan)

#     for beta_col in beta_result_column_ordering:
#         try:
#             fill_mask = np.logical_and(
#                 np.isnan(best_betas), ~np.isnan(result_df[beta_col])
#             )
#         except KeyError:
#             pass
#         else:
#             best_betas[fill_mask] = result_df[beta_col][fill_mask]
#     return best_betas


def prune_studies(results):
    pruned_results = []
    num_concordant_replications = []
    total_replications = []

    for (tissue, trait, gene), result_subset in tqdm(
        results.groupby(["tissue", "trait", "gene"])
    ):
        correlated_results = result_subset.loc[
            result_subset.mr_beta_choice == "beta_IVW_pca"
        ]
        if correlated_results.shape[0]:
            (
                correlated_results.sort_values("correlated_num_loci")
                .iloc[-1]
                .loc["study"]
            )
            pruned_results.append(
                correlated_results.sort_values("correlated_num_loci").iloc[-1]
            )
            # print(pruned_results)
            # break
        else:
            uncorrelated_results = result_subset.loc[
                result_subset.mr_beta_choice == "beta_IVW_uncorrelated"
            ]
            pruned_results.append(
                uncorrelated_results.sort_values("uncorrelated_num_loci").iloc[-1]
            )

        total_replications.append(result_subset.shape[0] - 1)
        if result_subset.shape[0] > 1:
            num_concordant_replications.append(
                np.equal(
                    np.sign(pruned_results[-1].mr_beta), np.sign(result_subset.mr_beta)
                ).sum()
                - 1
            )
        else:
            num_concordant_replications.append(0)

    pruned_results = pd.concat(pruned_results, axis=1).T.convert_dtypes()
    pruned_results["num_concordant_replications"] = num_concordant_replications
    pruned_results["total_replications"] = total_replications
    pruned_results["fraction_concordant_replications"] = (
        pruned_results["num_concordant_replications"]
        / pruned_results["total_replications"]
    )
    pruned_results["num_concordant_studies"] = (
        pruned_results.num_concordant_replications + 1
    )

    return pruned_results


def annotate_burgess_results(burgess_results: pd.DataFrame, output_fpath: Path | str):
    """
    Annotate Burgess analysis results

    Args:
        burgess_results: DataFrame with Burgess analysis results
        output_fpath: Path to save annotated results
    """
    burgess_results["neg_log10_pval"] = -np.log10(burgess_results.mr_beta_pval + 1e-300)
    burgess_results["mr_beta_qval"] = multipletests(
        burgess_results.mr_beta_pval, method="fdr_bh"
    )[1]
    burgess_results["neg_log10_qval"] = -np.log10(
        burgess_results["mr_beta_qval"] + 1e-300
    )

    burgess_results.to_csv(output_fpath, sep="\t", index=False)
    return burgess_results


def post_process_twosample_mr_results_single_gene(
    result_fpath: Path | str, output_fpath: Path | str, vtx_id: str, vtx_id_to_disease: Dict[str, str]
):
    """
    Post-process MR results for a single gene

    Args:
        result_fpath: Path to input results file
        output_fpath: Path to save processed results
        vtx_id: Vertex ID for the gene
    """
    results = pd.read_csv(result_fpath, sep="\t", low_memory=False)
    print(results.shape)
    results = results.drop_duplicates()
    print(results.shape)

    gene_names = []
    tissues = []
    for exposure in results.exposure:
        splat = exposure.split(" ")
        gene_names.append(splat[0])
        tissues.append(" ".join(splat[1:]).strip("()"))
    results["tissue"] = tissues
    results["gene_name"] = gene_names
    results["gene"] = gene_names

    results["vtx_id"] = vtx_id
    results["disease"] = [
        vtx_id_to_disease[vtx_id] for vtx_id in results.vtx_id
    ]
    results["trait"] = [outcome.split("||")[0].strip() for outcome in results.outcome]

    results["mr_beta"] = results.b
    results["mr_beta_se"] = results.se
    results["mr_beta_choice"] = results.method
    results["mr_beta_z"] = results["mr_beta"] / results["mr_beta_se"]
    results["mr_beta_pval"] = mr_models.beta_se_to_pval(
        results["mr_beta"], results["mr_beta_se"]
    )
    results["mr_beta_neg_log10_pval"] = -np.log10(results["mr_beta_pval"] + MIN_PVALUE)
    results["mr_beta_qval"] = fdrcorrection(results["mr_beta_pval"], method="i")[1]
    results["mr_beta_neg_log10_qval"] = -np.log10(results["mr_beta_qval"] + MIN_PVALUE)

    results["study"] = results["id.outcome"]

    results = results.dropna(
        subset=("mr_beta", "mr_beta_se", "mr_beta_pval")
    ).sort_values("mr_beta_pval")

    print(results.shape)
    results.to_csv(output_fpath, sep="\t", index=False)
    return results
