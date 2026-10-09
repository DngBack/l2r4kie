import importlib

import pytest


@pytest.mark.parametrize('name', ['data', 'model', 'train', 'confidence', 'eval', 'utils'])
def test_subpackage_imports(name):
    importlib.import_module(f'l2r4kie.{name}')
