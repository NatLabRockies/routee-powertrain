import logging as log
from pathlib import Path
from unittest import TestCase, skip

from tqdm import tqdm


import routee.powertrain as pt

this_dir = Path(__file__).parent

log.basicConfig(level=log.INFO)


class TestModelLibrary(TestCase):
    @skip(
        "This test is slow and not necessary for every build. "
        "Just test when the model library is updated."
    )
    def test_model_library(self):
        for mname in tqdm(pt.list_available_models()):
            model = pt.load_model(mname)
            self.assertIsNotNone(model)
