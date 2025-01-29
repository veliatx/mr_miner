"""Constants used throughout MR Miner."""


Z_95 = 3.919927969080109
PLINK_FPATH = 'plink'
DEFAULT_TRAIT_COLUMN = "trait_reported"
GENCODE_VERSION = "v42"
DEFAULT_POP = "EUR"
DEFAULT_GENOME_BUILD = "grch38"
DEFAULT_PROXY_R2_THRESHOLD = 0.9
DEFAULT_PROXY_DISTANCE_THRESHOLD_KB = 250
MIN_DET = 0.01
DEFAULT_PCA_VARIANCE_THRESHOLD = 0.99
DEFAULT_PLINK_THREADS = 64
MIN_PVALUE = 1e-300
UNIPROT_ID_MAPPING_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase//idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz"
DBSNP_SPDI_POS_OFFSET = 1
THOUSAND_GENOMES_POS_OFFSET = 0

DBSNP_CHROM_NAMESPACE = "refseq"
WORKING_CHROM_NAMESPACE = "ucsc"

# NCBI Assembly Report URL for GRCh38
NCBI_GRCH38_ASSEMBLY_REPORT_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/latest_assembly_versions/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_report.txt"

# LDLink API constants
LDLINK_API_TOKEN = 'c1b9ea431df9'
DEFAULT_MAX_GET_VARIANTS = 300
DEFAULT_MAX_POST_VARIANTS = 2500
DEFAULT_THREADS = 64

# Request handling constants
REQUEST_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.5

# Default paths
DEFAULT_OUTCOME_DATA_PATH = 'data/opentargets_variant_info_annotated_lead_only.tsv.gz'
