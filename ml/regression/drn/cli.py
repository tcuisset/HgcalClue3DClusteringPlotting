from lightning.pytorch.cli import LightningCLI

import sys
sys.path.append("../../..")
sys.path.append("/grid_mnt/vol_home/llr/cms/cuisset/hgcal/testbeam18/clue3d-dev/src/Plotting")

from ml.regression.drn.modules import *



def cli_main():
    cli = LightningCLI(RechitsDRNModule, RechitsDRNDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()