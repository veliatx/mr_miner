import collections
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import subprocess

from mr_miner.chrom_mapper import ChromMapper

logger = logging.getLogger(__name__)


def write_iterable(iterable: Iterable, output_fpath: Path | str):
    output_fpath = Path(output_fpath)
    if output_fpath.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(output_fpath, "wt") as output_file:
        output_file.writelines("\n".join(list(iterable)))


def load_iterable(input_fpath: Path | str):
    input_fpath = Path(input_fpath)
    if input_fpath.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(input_fpath, "rt") as input_file:
        iterable = [line.strip("\n") for line in input_file.readlines()]
    return iterable


def unmelt(
    df: pd.DataFrame, row_id_var: str, col_id_var: str, val_var: str, fill_val=0.0
) -> pd.DataFrame:
    unmelted_dict = collections.defaultdict(lambda: {})
    for row, col, val in iterate_cols(df, (row_id_var, col_id_var, val_var)):
        unmelted_dict[col][row] = val
    return pd.DataFrame(unmelted_dict).fillna(fill_val)


def unique_counts(dataframe: pd.DataFrame, normalize=False):
    if normalize:
        return {
            col: len(dataframe[col].unique()) / dataframe.shape[0]
            for col in dataframe.columns
        }
    else:
        return {col: len(dataframe[col].unique()) for col in dataframe.columns}


def convert_gtex_variant_id_to_spdi(variant_id: str, pos_offset: int = -1) -> str:
    chrom, pos, ref, alt, build = variant_id.split("_")
    return generate_spdi_str(chrom, int(pos) + pos_offset, ref, alt)


def trim_ensembl_gene(ensembl_gene_name: str):
    return ensembl_gene_name.split(".")[0]


def slugify_filename(fname, bad_chars="/", replacement_char="_"):
    for char in bad_chars:
        fname = fname.replace(char, replacement_char)
    return fname


def reflect_tri(tri_arr: np.ndarray) -> np.ndarray:
    """
    Takes a triangular (upper or lower) matrix and reflects it over the diagonal to generate a symmetric square matrix.
    """
    assert (
        tri_arr.shape[0] == tri_arr.shape[1]
    ), f"Require a square matrix but received shape {tri_arr.shape}!"

    sym_matrix = tri_arr + tri_arr.T
    sym_matrix[np.diag_indices(tri_arr.shape[0])] -= np.diag(tri_arr)

    return sym_matrix


def replace_df_values(df: pd.DataFrame, new_values: np.ndarray) -> pd.DataFrame:
    assert new_values.shape == df.shape
    return pd.DataFrame(new_values, index=df.index, columns=df.columns)


def plot_diag(ax):
    original_xmin, original_xmax = ax.get_xlim()
    original_ymin, original_ymax = ax.get_ylim()

    min_lim = min(original_xmin, original_ymin)
    max_lim = min(original_xmax, original_ymax)

    ax.plot((min_lim, max_lim), (min_lim, max_lim), linestyle="--", color="k")
    ax.set_xlim(original_xmin, original_xmax)
    ax.set_ylim(original_ymin, original_ymax)


def square_ax(ax, centered=False):
    if centered:
        max_lim = np.max(np.abs(np.hstack((ax.get_xlim(), ax.get_ylim()))))
        min_lim = -max_lim
    else:
        min_lim = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)


def binary_search(arr, query, find_exact=True):
    l = 0  # noqa: E741
    u = len(arr)

    while True:
        m = (u + l) // 2

        if arr[m] == query:
            return m
        else:
            if u - l <= 1:
                if find_exact:
                    return -1
                else:
                    return m

            if arr[m] > query:
                u = m
            elif arr[m] < query:
                l = m  # noqa: E741


def decompose_spdi(spdi: str, sep=":"):
    return spdi.split(":")


def compare_sets(a: Iterable, b: Iterable) -> Dict[str, int | float]:
    a = set(a)
    b = set(b)

    return {
        "only_a": len(a.difference(b)),
        "only_b": len(b.difference(a)),
        "common": len(a.intersection(b)),
    }


def replace_index(df, new_index_col):
    old_index = df.index
    df.index = df[new_index_col]
    df = df.drop(new_index_col, axis=1)
    # print(df)
    df[old_index.name] = old_index
    return df


# class Logger:
#     """
#     Minimalist re-implementation of the Logger class for use in Jupyter notebooks.
#     """
#     def __init__(self, fpath='', new=False):
#         self.fpath = fpath
#         if new:
#             try:
#                 os.remove(self.fpath)
#             except FileNotFoundError:
#                 pass

#     def info(self, msg):
#         print_string = f'{datetime.datetime.strftime(datetime.datetime.now(), format="%D %H:%M:%S")}\t{msg}'
#         print(print_string)
#         if self.fpath:
#             with open(self.fpath, 'at') as log_file:
#                 log_file.write(print_string + '\n')


def setup_logger(log_fpath: str, verbosity: int = logging.INFO) -> None:
    """
    Set up the logger.

    Args:
        log_fpath (str): The path to the log file.
        verbosity (int, optional): The level of logging. Defaults to logging.INFO.
    """
    # global logger
    logger = logging.getLogger()
    logger.handlers = []

    logger.setLevel(logging.DEBUG)

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(verbosity)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)

    # create file handler if specified
    if log_fpath:
        fh = logging.StreamHandler(gzip.open(log_fpath, mode="wt", encoding="utf-8"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Logger set up at level %s and writing to %s.", verbosity, log_fpath)

    return logger


def find_all_attributes(
    gff_df: pd.DataFrame, attribute_column: str = "attributes"
) -> Set[str]:
    """
    Finds all unique attributes in a specified column of a GFF DataFrame.

    Args:
        gff_df (pd.DataFrame): The GFF DataFrame.
        attribute_column (str, optional): The column name to search for attributes. Defaults to 'attribute'.

    Returns:
        Set[str]: A set of all unique attributes found.
    """
    all_attrs = set([])
    for attribute_dict in tqdm(gff_df[attribute_column]):
        all_attrs.update(attribute_dict.keys())

    return all_attrs


def parse_gff_attributes(
    attribute_string, kv_pair_separator=";", list_separator=",", assignment_operator="="
):
    attribute_dict = {}
    kv_pairs = attribute_string.split(kv_pair_separator)
    for kv in kv_pairs:
        if not len(kv):
            continue
        k, v = kv.split(assignment_operator)

        v_splat = v.split(list_separator)

        if len(v_splat) == 1:
            attribute_dict[k] = v
        else:
            attribute_dict[k] = v_splat

    return attribute_dict


def sparsify_dicts(
    dense_df: pd.DataFrame, columns: List[str], null_value: str = ""
) -> pd.DataFrame:
    """
    Converts dense dictionary columns in a DataFrame to sparse format.

    Args:
        dense_df (pd.DataFrame): The dense DataFrame.
        columns (List[str]): The columns to sparsify.
        null_value (str, optional): The value to use for missing data. Defaults to ''.

    Returns:
        pd.DataFrame: The DataFrame with sparsified columns.
    """
    new_cols = set([])
    for col in columns:
        new_cols.update(find_all_attributes(dense_df, col))

    new_columns = {col: {} for col in new_cols}

    for col in columns:
        for row_idx, attrs in tqdm(dense_df[col].items()):
            for k, v in attrs.items():
                new_columns[k][row_idx] = v

    sparse_df = dense_df.copy()
    for col, col_values in new_columns.items():
        sparse_df[col] = pd.Series(col_values)
        sparse_df[col] = sparse_df[col].fillna(null_value)

    return sparse_df


def find_all_tags(gff_df: pd.DataFrame, tag_column: str, sep: str = ",") -> Set[str]:
    """
    Finds all unique tags in a specified column of a DataFrame, where tags are separated by a specified separator.

    Args:
        gff_df (pd.DataFrame): The DataFrame to search.
        tag_column (str): The column name to search for tags.
        sep (str, optional): The separator used between tags. Defaults to ','.

    Returns:
        Set[str]: A set of all unique tags found.
    """
    all_tags = set([])
    for tag_entry in tqdm(gff_df[tag_column]):
        all_tags.update(tag_entry.split(sep))
    return all_tags


def sparsify_lists(
    dense_df: pd.DataFrame, columns: List[str], sep: str = ","
) -> pd.DataFrame:
    """
    Converts dense list columns in a DataFrame to sparse format, where lists are separated by a specified separator.

    Args:
        dense_df (pd.DataFrame): The dense DataFrame.
        columns (List[str]): The columns to sparsify.
        sep (str, optional): The separator used between list items. Defaults to ','.

    Returns:
        pd.DataFrame: The DataFrame with sparsified columns.
    """
    new_cols = set([])

    for col in columns:
        new_cols.update(find_all_tags(dense_df, col, sep=sep))

    new_columns = {col: {} for col in new_cols}

    for col in columns:
        for row_idx, cell_value in tqdm(zip(dense_df.index, dense_df[col])):
            tags = cell_value.split(sep)
            for tag in tags:
                new_columns[tag][row_idx] = True

    col_series = {}
    for col, col_values in new_columns.items():
        col_series[col] = pd.Series(col_values, dtype=bool)

    null_value = False
    new_col_df = pd.DataFrame(col_series).fillna(null_value)

    return pd.concat((dense_df, new_col_df), axis=1)


def load_genome(genome_fasta_fpath: str):
    """
    Load genome sequence from FASTA file

    Args:
        genome_fasta_fpath: Path to genome FASTA file

    Returns:
        Dict mapping chromosome names to sequences
    """
    genome = {}
    with open(genome_fasta_fpath) as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    genome[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            genome[current_id] = ''.join(current_seq)
    return genome


def parse_genebass_marker_id(
    marker_id, chrom_mapper: ChromMapper, source_chrom_dialect="ucsc", destination_chrom_dialect="ucsc"
):
    chrom, everything_else = marker_id.split(":")

    if source_chrom_dialect != destination_chrom_dialect:
        chrom = chrom_mapper.translate_chrom_name(
            chrom, source_chrom_dialect, destination_chrom_dialect
        )

    pos, refalt = everything_else.split("_")
    pos = int(pos)
    ref, alt = refalt.split("/")

    pos, ref, alt = right_shift_variant_for_vep(pos, ref, alt)

    return chrom, pos, ref, alt


def populate_genebass_spdi(genebass_df, sep=":", add_locus_columns=True):
    spdis_by_marker_id = {}
    spdis = []

    if add_locus_columns:
        chroms = []
        starts = []
        ends = []

        bed_info_by_marker_id = {}

    for marker_id in genebass_df.markerID:
        if marker_id in spdis_by_marker_id:
            this_spdi = spdis_by_marker_id[marker_id]

        else:
            chrom, everything_else = marker_id.split(":")
            # chrom = cm.translate_chrom_name(chrom, 'ucsc', 'plain')
            pos, refalt = everything_else.split("_")
            pos = int(pos)
            ref, alt = refalt.split("/")
            pos, ref, alt = right_shift_variant_for_vep(pos, ref, alt)

            this_spdi = ":".join((chrom, str(pos - 1), ref, alt))
            spdis_by_marker_id[marker_id] = this_spdi

        spdis.append(this_spdi)

        if add_locus_columns:
            if marker_id in bed_info_by_marker_id:
                this_chrom, this_start, this_end = bed_info_by_marker_id[marker_id]
            else:
                this_chrom = chrom
                this_start = pos
                this_end = pos + len(ref)
                bed_info_by_marker_id[marker_id] = (this_chrom, this_start, this_end)

            chroms.append(this_chrom)
            starts.append(this_start)
            ends.append(this_end)

    genebass_df["spdi"] = spdis

    if add_locus_columns:
        genebass_df["chrom"] = chroms
        genebass_df["start"] = starts
        genebass_df["end"] = ends


def construct_vep_input_line(chrom, pos, ref, alt) -> Tuple:
    pos = int(pos)
    pos, ref, alt = right_shift_variant_for_vep(pos, ref, alt)

    if ref == "-":  # insertion
        end = pos
        start = end + 1
    elif alt == "-":  # deletion
        start = pos
        end = start + len(ref) - 1  # intervals are fully closed
    else:  # SNV
        assert len(ref) == len(
            alt
        ), f"Length of REF ({len(ref)}) doesn't equal length of ALT ({len(alt)})!"
        interval_len = len(ref)
        start = pos
        end = start + interval_len - 1

    return chrom, start, end, f"{ref}/{alt}", "+"


def right_shift_variant_for_vep(pos, ref, alt):
    i = 0
    while i < min(len(ref), len(alt)):
        if ref[i] == alt[i]:
            i += 1
        else:
            break

    pos += i
    if i == len(ref):
        ref = "-"
    else:
        ref = ref[i:]

    if i == len(alt):
        alt = "-"
    else:
        alt = alt[i:]

    return pos, ref, alt


def construct_vep_input_line_from_genebass_marker_id(marker_id: str) -> Tuple:
    return construct_vep_input_line(*parse_genebass_marker_id(marker_id))


def construct_vep_input_line_from_spdi(spdi: str) -> Tuple:
    return construct_vep_input_line(*(spdi.split(":")))


def iterate_cols(df, col_subset, preface_with_index=False, tqdm_desc=""):
    if tqdm_desc:
        for tup in tqdm(
            zip(df.index, *[df[col] for col in col_subset]),
            total=df.shape[0],
            desc=tqdm_desc,
        ):
            if preface_with_index:
                yield tup
            else:
                yield tup[1:]
    else:
        for tup in zip(df.index, *[df[col] for col in col_subset]):
            if preface_with_index:
                yield tup
            else:
                yield tup[1:]


def compute_column_map(df, source_col, target_col):
    return pd.Series(
        {
            source_col_value: source_col_data[target_col].unique()
            for source_col_value, source_col_data in df.groupby(source_col)
        }
    )


def compute_column_cardinality(df, source_col, target_col):
    return pd.Series(
        {
            source_col_value: len(source_col_data[target_col].unique())
            for source_col_value, source_col_data in df.groupby(source_col)
        }
    )


# def compute_cardinality_matrix(df):
#     vars = df.columns
#     for source_var, target_var in tqdm(list(itertools.product(vars, vars))):
#         if source_var == target_var:
#             cardinalities.loc[source_var, target_var] = (
#                 df[source_var].value_counts().mean()
#             )
#         else:
#             cardinalities.loc[source_var, target_var] = compute_column_cardinality(
#                 df, source_var, target_var
#             ).mean()

#     return cardinalities


def my_in1d(arr1, arr2):
    """
    Just like numpy.in1d() but much faster (for some reason)
    """
    arr2 = set(arr2)
    return np.array([element in arr2 for element in arr1])




def generate_spdi_str(chrom, pos, ref, alt, sep=":"):
    # Per ensembl, the position field in SPDI points to the genomic coordinate (using 1-based indexing)
    # of the base immediately before the start of the reference allele. Adjust the pos accordingly before
    # passing to this function.
    return sep.join((str(chrom), str(pos), str(ref), str(alt)))


def generate_ref_alt_str(chrom, pos, ref, alt):
    return f"{chrom}-{pos}-{ref}-{alt}"


def sterilize_default_dict(dd):
    return {k: v for k, v in dd.items()}


def convert_dtypes_correctly(df: pd.DataFrame):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

def generate_spdi_string(chrom, pos, ref, alt, sep=":"):
    return sep.join((str(chrom), str(pos), str(ref), str(alt)))


def save_pickle(obj: Any, output_fpath: Path | str, protocol: int = 4) -> None:
    """
    Save object to pickle file

    Args:
        obj: Object to save
        output_fpath: Path to save pickle file
        protocol: Pickle protocol version
    """
    with open(output_fpath, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def load_pickle(pickle_fpath: Path | str) -> Any:
    """
    Load object from pickle file

    Args:
        pickle_fpath: Path to pickle file
    """
    with open(pickle_fpath, "rb") as f:
        return pickle.load(f)


def check_command_exists(command: str, package_name: str = None) -> bool:
    """Check if a command exists and provide installation guidance if missing.
    
    Args:
        command: Name of the command to check
        package_name: Optional package name if different from command
        
    Returns:
        bool: True if command exists, False if not
    """
    try:
        subprocess.run(['which', command], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        package = package_name or command
        logger.error(f"Required command '{command}' not found. Please install it with:")
        logger.error(f"  sudo apt-get install {package}")
        return False


def check_required_commands(commands: dict[str, str | None]) -> None:
    """Check multiple required commands exist.
    
    Args:
        commands: Dict mapping commands to their package names (or None if same as command)
        
    Raises:
        RuntimeError: If any required commands are missing
    """
    missing = []
    for cmd, pkg in commands.items():
        if not check_command_exists(cmd, pkg):
            missing.append(cmd)
    
    if missing:
        raise RuntimeError(f"Missing required commands: {', '.join(missing)}")
