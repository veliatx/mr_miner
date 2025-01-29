import logging
from typing import Dict
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import scipy.stats
from sklearn.decomposition import PCA
from numpy.linalg import inv, cholesky

from mr_miner.constants import DEFAULT_PCA_VARIANCE_THRESHOLD

logger = logging.getLogger(__name__)


def pick_best_betas(
    result_df: pd.DataFrame,
    beta_result_column_ordering: tuple = (
        "beta_IVW_pca",
        "beta_IVW_cholesky",
        "beta_IVW_uncorrelated",
    ),
) -> np.ndarray:
    best_betas = np.full(shape=result_df.shape[0], fill_value=np.nan)

    for beta_col in beta_result_column_ordering:
        try:
            fill_mask = np.logical_and(
                np.isnan(best_betas), ~np.isnan(result_df[beta_col])
            )
        except KeyError:
            pass
        else:
            best_betas[fill_mask] = result_df[beta_col][fill_mask]
    return best_betas


def beta_se_to_pval(betas: np.ndarray, beta_ses: np.ndarray) -> np.ndarray:
    right_tailed_p = scipy.stats.norm(loc=0, scale=beta_ses).sf(betas)
    left_tailed_p = scipy.stats.norm(loc=0, scale=beta_ses).cdf(betas)
    return np.minimum(left_tailed_p, right_tailed_p) * 2


class MRModel:
    def __init__(self, cache_dir_path: Path | str | None = None):
        """
        Initialize MR model

        Args:
            cache_dir_path: Optional directory to cache results
        """
        self.cache_dir_path = Path(cache_dir_path) if cache_dir_path else None
        self._params: Dict[str, float] = {}

        if self.cache_dir_path:
            self.cache_dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def params(self) -> Dict[str, float]:
        return self._params

    @staticmethod
    def compute_resid_se(
        linear_model_results: sm.regression.linear_model.RegressionResults
    ) -> float:
        rss = linear_model_results.model.weights @ (linear_model_results.resid**2)
        df = linear_model_results.df_resid

        if df == 0:
            return np.nan
        return np.sqrt(rss / df)


class BurgessUncorrelated(MRModel):
    def __init__(self):
        super().__init__()

    def fit(self, betas_df: pd.DataFrame) -> None:
        y = np.row_stack(betas_df.outcome_beta.values)
        X = np.row_stack(betas_df.exposure_beta.values)
        w = betas_df.outcome_beta_se**-2

        # Running Weighted Least Squares (WLS)
        wls = sm.WLS(y, X, weights=w, hasconst=False)
        res = wls.fit()

        betaIVW_uncorrelated = res.params[0]
        se_IVW_fixed = res.bse[0] / self.compute_resid_se(res)

        se_IVW_random = min(se_IVW_fixed, 1)

        self._params = {
            "beta_IVW_uncorrelated": betaIVW_uncorrelated,
            "beta_se_IVW_uncorrelated_fixed": se_IVW_fixed,
            "beta_se_IVW_uncorrelated_random": se_IVW_random,
        }


class WaldEstimator(MRModel):
    def __init__(self):
        super().__init__()

    def fit(self, betas_df: pd.DataFrame) -> None:
        assert (
            betas_df.shape[0] == 1
        ), "Wald estimation can only be performed for single variants!"

        self._params = {
            "beta_IVW_uncorrelated": betas_df.outcome_beta.iloc[0]
            / betas_df.exposure_beta.iloc[0],
            "beta_se_IVW_uncorrelated_fixed": self.compute_wald_se(
                beta_Y=betas_df.outcome_beta.iloc[0],
                beta_X=betas_df.exposure_beta.iloc[0],
                se_Y=betas_df.outcome_beta_se.iloc[0],
                se_X=betas_df.exposure_beta_se.iloc[0],
            ),
        }

    @staticmethod
    def compute_wald_se(beta_Y: float, beta_X: float, se_Y: float, se_X: float) -> float:
        """
        Compute the standard error of the estimated beta for a Wald estimate.

        Parameters:
            beta_Y (float): Effect size of the variant on the outcome.
            beta_X (float): Effect size of the variant on the exposure.
            se_Y (float): Standard error of beta_Y.
            se_X (float): Standard error of beta_X.

        Returns:
            float: Standard error of the Wald estimate.
        """
        if beta_X == 0:
            raise ValueError("beta_X cannot be zero to avoid division by zero.")

        var_Y = se_Y**2
        var_X = se_X**2

        # Apply the delta method formula
        wald_variance = var_Y / (beta_X**2) + (beta_Y**2 * var_X) / (beta_X**4)
        wald_se = np.sqrt(wald_variance)

        return wald_se


class BurgessCorrelated(MRModel):
    def __init__(self):
        super().__init__()

    def fit(self, betas_df: pd.DataFrame, rho_df: pd.DataFrame) -> None:
        sebetaYG = betas_df.outcome_beta_se
        betaXG = betas_df.exposure_beta
        betaYG = betas_df.outcome_beta

        omega = np.outer(sebetaYG, sebetaYG) * rho_df
        omega_inv = inv(omega)

        beta_IVWcorrel = (
            1 / (betaXG.values.T @ omega_inv @ betaXG) * betaXG.T @ omega_inv @ betaYG
        )
        se_IVWcorrel_fixed = np.sqrt(1 / (betaXG.T @ omega_inv @ betaXG))

        resid = betaYG - beta_IVWcorrel * betaXG
        se_IVWcorrel_random = np.sqrt(1 / (betaXG.T @ omega_inv @ betaXG)) * max(
            np.sqrt(resid.T @ omega_inv @ resid / (len(betaXG) - 1)), 1
        )

        self._params = {
            "beta_IVW_correlated": beta_IVWcorrel,
            "beta_se_IVW_correlated_fixed": se_IVWcorrel_fixed,
            "beta_se_IVW_correlated_random": se_IVWcorrel_random,
        }


class BurgessCholesky(MRModel):
    def __init__(self):
        super().__init__()

    def fit(self, betas_df: pd.DataFrame, rho_df: pd.DataFrame) -> None:
        sebetaYG = betas_df.outcome_beta_se
        betaXG = betas_df.exposure_beta
        betaYG = betas_df.outcome_beta

        omega = np.outer(sebetaYG, sebetaYG) * rho_df
        inv_cholesky = inv(cholesky(omega))
        c_betaXG = inv_cholesky @ betaXG
        c_betaYG = inv_cholesky @ betaYG

        res = sm.OLS(c_betaYG, c_betaXG).fit()
        beta_IVWcorrel = res.params[0]
        se_IVWcorrel_fixed = np.sqrt(1 / (betaXG @ inv(omega) @ betaXG))
        se_IVWcorrel_random = se_IVWcorrel_fixed * max(self.compute_resid_se(res), 1)

        self._params = {
            "beta_IVW_cholesky": beta_IVWcorrel,
            "beta_se_IVW_cholesky_fixed": se_IVWcorrel_fixed,
            "beta_se_IVW_cholesky_random": se_IVWcorrel_random,
        }


class BurgessPCA(MRModel):
    def __init__(
        self,
        pca_variance_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD,
    ):
        super().__init__()
        self.pca_variance_threshold = pca_variance_threshold

    def fit(
        self, betas_df: pd.DataFrame, rho_df: pd.DataFrame
    ) -> Dict[str, float]:
        assert betas_df.shape[0] == rho_df.shape[0] == rho_df.shape[1]
        if betas_df.shape[0] < 2:
            raise ValueError("Must have more than one data point to run PCA model!")

        sebetaYG = betas_df.outcome_beta_se
        betaXG = betas_df.exposure_beta
        betaYG = betas_df.outcome_beta

        b = betaXG / sebetaYG
        phi = np.outer(b, b) * rho_df
        phi_pca = PCA().fit(phi)

        cumulative_ev = 0
        for k, evr in enumerate(phi_pca.explained_variance_ratio_):
            if np.isnan(evr):
                raise ValueError(
                    f"Encountered NaN in explained variance ratios for component {k}."
                )
            cumulative_ev += evr
            if cumulative_ev >= self.pca_variance_threshold:
                break
        k += 1

        betaXG0 = betaXG @ phi_pca.components_[:k].T
        betaYG0 = betaYG @ phi_pca.components_[:k].T

        omega = np.outer(sebetaYG, sebetaYG) * rho_df
        pc_omega = phi_pca.components_[:k] @ omega @ phi_pca.components_[:k].T

        inv_pc_omega = inv(pc_omega)

        beta_ivw_pc = (
            1 / (betaXG0 @ inv_pc_omega @ betaXG0) * betaXG0 @ inv_pc_omega @ betaYG0
        )

        se_ivw_pca_fixed = np.sqrt(1 / (betaXG0 @ inv_pc_omega @ betaXG0))

        self._params = {
            "beta_IVW_pca": beta_ivw_pc,
            "beta_se_IVW_pca_fixed": se_ivw_pca_fixed,
        }
