import os
import argparse
import sys
import inspect

import uproot
import awkward as ak
import hist
import pandas as pd

from HistogramLib.histogram import *
from HistogramLib.store import * 
from dataframe import *
import custom_hists

parser = argparse.ArgumentParser(description="Fill histograms from CLUE_clusters.root files for plotting. Writes pickled boost-histograms")
parser.add_argument("--input", dest="input_file", default='ClusteringAnalysis/CLUE_clusters_single.root',
    help="Complete path to CLUE output, usually named CLUE_clusters.root")
parser.add_argument("--output", dest="output_directory",
    default='/home/llr/cms/cuisset/hgcal/testbeam18/clue3d-dev/src/plots/cache/',
    help="Path to output histograms. Will write in clue_param/datatype/hists.shelf")
parser.add_argument("--datatype", dest="datatype", default="data",
    help="Can be data, sim_proton, sim_noproton")
parser.add_argument("--clue-params", dest='clue_params', default="default",
    help="CLUE3D parameters name")
args = parser.parse_args()

##### DEBUG
args.input_file = "/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v3/default/data/CLUE_clusters.root"
args.output_directory = "/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v3/"


hist_dict = {}

# Create inside shelf all histogram classes located in module custom_hists.py
for name in dir(custom_hists):
    potentialClass = getattr(custom_hists, name)
    # need to exclude the case where MyHistogram gets selected
    if isinstance(potentialClass, type) and issubclass(potentialClass, MyHistogram) and potentialClass is not MyHistogram:
        hist_dict[potentialClass.__name__] = potentialClass()

with HistogramStore(args.output_directory, 'c', makedirs=True) as store:
    try:
        for (array, report) in uproot.iterate(args.input_file + ":clusters", step_size="50MB", library="ak", report=True):
            print("Processing events [" + str(report.start) + ", " + str(report.stop) + "[")

            comp = DataframeComputations(array)
            for histogram in hist_dict.values():
                histogram.loadFromComp(comp)
            del comp

    except IndexError as e:
        print("WARNING : an IndexError exception ocurred. This can happen for improperly closed ROOT files.")
        print("WARNING : the last batch of entries may not have been processed, but the histograms will be written anyway")
        print("The exception was : ")
        print(e)
    
    print("Writing histograms to file...")
    shelf = store.getShelf(ShelfId(args.clue_params, args.datatype))
    for h_name, h in hist_dict.items():
        shelf[h_name] = h
    print("Syncing...")
print("Done")

# if hist_dict["rechits_position"].empty():
#     print("Result histogram is empty")
#     sys.exit(-1)


