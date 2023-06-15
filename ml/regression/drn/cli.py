from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

import sys
sys.path.append("../../..")
sys.path.append("/grid_mnt/vol_home/llr/cms/cuisset/hgcal/testbeam18/clue3d-dev/src/Plotting")

from ml.regression.drn.modules import *
from ml.regression.drn.callbacks import SigmaOverECallback

class DRNCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(SigmaOverECallback, "sigma_over_e_callback")
        parser.link_arguments("data.reader", "sigma_over_e_callback.overlaySigmaOverEResults",
            lambda reader : [reader.loadSigmaOverEResults("rechits")], apply_on="instantiate")



if __name__ == "__main__":
    cli = DRNCLI(DRNModule, DRNDataModule)