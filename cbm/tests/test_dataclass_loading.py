import pytest
import torch
from transformers import BertTokenizer

from ..data.cebab import CEBaBDataset
from ..data.imdb import IMDBDataset


@pytest.mark.parametrize("variant,expected_min_len,expect_extra", [
    ("pure", 1, False),
    ("aug", 1, True),
])
def test_cebab_loading(variant, expected_min_len, expect_extra):
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = CEBaBDataset('train', tok, 16, variant=variant)
    assert len(ds) >= expected_min_len
    item = ds[0]
    assert 'input_ids' in item and 'attention_mask' in item and 'label' in item
    assert 'concept_labels' in item
    assert item['concept_labels'].numel() == (10 if expect_extra else 4)


@pytest.mark.parametrize("variant,expected_min_len,expect_extra", [
    ("manual", 1, False),
    ("aug_gen", 1, True),
])
def test_imdb_loading(variant, expected_min_len, expect_extra):
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = IMDBDataset('train', tok, 16, variant=variant)
    assert len(ds) >= expected_min_len
    item = ds[0]
    assert 'input_ids' in item and 'attention_mask' in item and 'label' in item
    assert 'concept_labels' in item
    assert item['concept_labels'].numel() == (8 if expect_extra else 4)


