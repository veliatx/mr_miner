"""Tests for utility functions."""
import pytest

from mr_miner.utilities import check_required_commands, iterate_cols


def test_check_required_commands():
    required_commands = {
        'python': None,
    }
    check_required_commands(required_commands)
    
    required_commands['nonexistentcmd'] = 'alternativecmd'
    with pytest.raises(RuntimeError, match="Missing required commands: nonexistentcmd"):
        check_required_commands(required_commands)


def test_iterate_cols():
    import pandas as pd

    data = {
        2: ['gene', 'gene', 'transcript'],
        8: [
            'gene_id=GENE1;Name=GeneOne',
            'gene_id=GENE2;Name=GeneTwo',
            'transcript_id=TRANS1;Name=TranscriptOne'
        ]
    }
    df = pd.DataFrame(data)

    results = list(iterate_cols(df, [2, 8], preface_with_index=True))
    assert results == [
        (0, 'gene', 'gene_id=GENE1;Name=GeneOne'),
        (1, 'gene', 'gene_id=GENE2;Name=GeneTwo'),
        (2, 'transcript', 'transcript_id=TRANS1;Name=TranscriptOne')
    ] 