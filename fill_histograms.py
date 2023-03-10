import os
import argparse

import uproot
import hist

from HistogramLib.histogram import MyHistogram
from HistogramLib.store import HistogramStore, ShelfId
from hists.dataframe import *
from hists.store import *
from hists import custom_hists

default_tag = 'v4'

parser = argparse.ArgumentParser(description="Fill histograms from CLUE_clusters.root files for plotting. Writes pickled boost-histograms")
parser.add_argument("--data-dir", dest='data_dir',
    default=os.path.join('/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/', default_tag),
    help="The path to the directory where all the data is, with the tag included, but with clue_params/datatype exluded",
)
parser.add_argument("--force-input-file", dest='force_input_file',
    default=None,
    help="Complete path to CLUE3D output file (for testing)")
parser.add_argument("--force-output-directory", dest='force_output_directory',
    default=None,
    help="Complete path to folder where to store histograms (as a python shelve database, files hists.shelve.*) (for testing)")

parser.add_argument("--datatype", dest="datatype", default="data",
    help="Can be data, sim_proton, sim_noproton")
parser.add_argument("--clue-params", dest='clue_params', default="single-file",
    help="CLUE3D parameters name")
args = parser.parse_args()

if args.force_input_file is not None:
    input_file = args.force_input_file
else:
    input_file = os.path.join(args.data_dir, args.clue_params, args.datatype, 'CLUE_clusters.root')

if args.force_output_directory is not None:
    output_dir = args.force_output_directory
else:
    output_dir = args.data_dir

hist_dict = {}

# Create inside shelf all histogram classes located in module custom_hists.py
for name in dir(custom_hists):
    potentialClass = getattr(custom_hists, name)
    # need to exclude the case where MyHistogram gets selected
    if isinstance(potentialClass, type) and issubclass(potentialClass, MyHistogram) and potentialClass is not MyHistogram:
        hist_dict[potentialClass.__name__] = potentialClass()

print("Opening input file", flush=True)
store = HistogramStore(output_dir, HistogramId)
try:
    # step_size of 50MB stranslates to about 5GB of memory usage by python, and about 4k events at a time
    # 500MB leads to 50k events at a time, and memory usage of of around 10 GB (partly due to big histograms)
    for (array, report) in uproot.iterate(input_file + ":clusters", step_size="500MB", library="ak", report=True):
        print("Processing events [" + str(report.start) + ", " + str(report.stop) + "[", flush=True)

        comp = DataframeComputations(array)
        for histogram in hist_dict.values():
            histogram.loadFromComp(comp)
        del comp

except IndexError as e:
    print("WARNING : an IndexError exception ocurred. This can happen for improperly closed ROOT files.")
    print("WARNING : the last batch of entries may not have been processed, but the histograms will be written anyway")
    print("The exception was : ")
    print(e)

print("Writing histograms to file...", flush=True)
for h_name, h in hist_dict.items():
    print(h_name, flush=True)
    histId = HistogramId(clueParamName=args.clue_params, datatype=args.datatype, histName=h_name)
    store.save(histId, h, makedirs=True)
print("Done")

# if hist_dict["rechits_position"].empty():
#     print("Result histogram is empty")
#     sys.exit(-1)


