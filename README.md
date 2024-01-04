# Plotting code for CLUE3D clustering analysis study (HGCAL/CMS)
## Quickstart : How to access an already running plot server
There might be a bokeh server already running on llruicms on port 50008
in this case to access it : 
 - if on LLR network, open <http://llruicms01:5008>
 - if not on LLR network, use ssh to forward port 5008 : `ssh -NL 5008:llruicms01:5008 <your_username>@llrgate01.in2p3.fr` then open <http://localhost:5008>
Then choose either rechits, cluster2D or cluster3D

## How to setup
You need to install all the relevant packages using conda (see [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)). The environment is called `clustering-analysis`

    conda env create -f environment.yml

Then you might want to activate it : 

    conda activate clustering-analysis

## How to run
### Step 1 : run CLUE3D
This is done in HgcalClue3DClusteringAnalysis repo
output files are stored in `/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/`

### Step 2 : Run cython
Only needed once, when you change python version.
~~~bash
cd hists
cythonize -3 -i dataframe_cython.pyx
~~~

### Step 3 : fill histograms
This fills multidimensional histograms using pandas from the output of CLUE3D (`CLUE_clusters.root` files)
`--data-dir` is the working directory for input and output files. `--clue-params` is a tag name for input ntuples (usually just set to `cmssw`, as in CLUE3D parameters taken from CMSSW at the time of this work)
~~~bash
TAG_VERSION=v44

python fill_histograms.py --data-dir=/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/$TAG_VERSION/  --clue-params=cmssw --datatype=data --save-metadata
python fill_histograms.py --data-dir=/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/$TAG_VERSION/  --clue-params=cmssw --datatype=sim_proton_v46_patchMIP
~~~

### Step 4 : plotting
Use all the notebooks scattered in the repository. Don't forget to update the input folder at the beginning of each notebook to use the input samples you want

## Code organization
### Top-level python scripts
`fill_histograms.py` is the main histogram-making script you want to use.

The `rechits.py`, `clusters*.py`, `global.py` are old scripts designed to run a Bokeh server for interactive plot visualization. They are not used anymore, notebooks are used instead (they might still work but are a bit complicated to setup)

`dash_event_visualizer.py` runs a Dash server for event visualization. It is used for the website for interactive event display at <https://hgcal-tb18-clue3d-visualization.web.cern.ch/>


### HistogramLib
Common code for bokeh and histogram projection, not specific to this analysis

### hists
Where all histograms are defined and loaded from pandas dataframe. TO create a new histogram, you should add the histogram itself in `hist/custom_hists.py` and the code to make the dataframe used for filling the histogram in `hist/dataframe.py`

### fill_histogram.py
main script to fill the histograms

### event_visualizer
Code for event visualization using Plotly and Dash (3D event displays as well as 2D layer views, both in notebooks and in web browser)

### ml
Study for energy regression using graph neural network

### bokeh_apps
bokeh code common to all endpoints

### rechits.py, cluster2D.py, cluster3D.py
endpoints for bokeh server, meant to be run using `bokeh serve`