import gzip
from pathlib import Path
from typing import Collection, Dict, Iterable

from mr_miner.plink_wrapper import PlinkWrapper


class VariantProxies:
    """
    Wraps a PlinkWrapper to find variant proxies with given parameters for min R2 and distance.
    Caches the results in RAM and on disk.
    """

    def __init__(
        self,
        plink_wrapper: PlinkWrapper,
        cache_path: Path | str,
        r2_threshold: float,
        distance_threshold_kb: float,
    ) -> None:
        self.plink_wrapper = plink_wrapper
        self.cache_path = Path(cache_path)
        self.r2_threshold = r2_threshold
        self.distance_threshold_kb = distance_threshold_kb

        self.proxies: Dict[str, Collection[str]] = {}
        if self.cache_path:
            stem_suffix = f"_R2_{self.r2_threshold}_{self.distance_threshold_kb}KB"
            self.cache_fpath = self.cache_path.joinpath(
                f"variant_proxies_{stem_suffix}.txt.gz"
            )
            self._load_from_cache()
        else:
            self.cache_fpath = None

    def _load_from_cache(self) -> None:
        if not self.cache_fpath.exists():
            return
        try:
            with gzip.open(self.cache_fpath, "rt") as cache_file:
                for line in cache_file:
                    splat = line.strip().split("\t")
                    proxy_source_fpath = splat[0]
                    if len(splat) == 1:
                        proxy_targets_fpaths = []
                    else:
                        proxy_targets_fpaths = splat[1:]
                    self.proxies[proxy_source_fpath] = proxy_targets_fpaths
        except FileNotFoundError:
            pass

    def _rewrite_cache(self) -> None:
        self._write_to_cache(self.proxies, overwrite=True)

    def _write_to_cache(
        self, proxy_dict: Dict[str, Collection[str]], overwrite: bool = False
    ) -> None:
        mode = "wt" if overwrite else "at"
        with gzip.open(self.cache_fpath, mode) as cache_file:
            for proxy_source_fpath, proxy_targets_fpaths in proxy_dict.items():
                cache_file.write(
                    "\t".join([proxy_source_fpath] + list(proxy_targets_fpaths)) + "\n"
                )

    def _query_plink(self, query_rsids: Iterable[str]) -> Dict[str, Collection[str]]:
        return self.plink_wrapper.get_proxy_variants(
            query_rsids=query_rsids,
            r2_threshold=self.r2_threshold,
            distance_kbp=self.distance_threshold_kb,
        )

    def get_variant_proxies(
        self, query_rsids: Iterable[str]
    ) -> Dict[str, Collection[str]]:
        query_results: Dict[str, Collection[str]] = {}
        plink_query_queue: set = set()

        for rsid in query_rsids:
            if rsid in self.proxies:
                query_results[rsid] = self.proxies[rsid]
            else:
                plink_query_queue.add(rsid)

        if plink_query_queue:
            plink_results = self._query_plink(plink_query_queue)

            new_results: Dict[str, Collection[str]] = {}
            for rsid in plink_query_queue:
                if rsid in plink_results:
                    new_results[rsid] = plink_results[rsid]
                else:
                    new_results[rsid] = []

            self.proxies.update(new_results)
            self._write_to_cache(new_results)

        return query_results