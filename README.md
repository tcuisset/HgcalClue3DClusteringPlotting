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

### Step 2 : fill histograms
This fills multidimensional histograms using pandas from the output of CLUE3D (`CLUE_clusters.root` files)

    python fill_histograms.py --data-dir=/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v3/  --datatype=data --clue-params=default

### Step 3 : run the bokeh server

    conda activate clustering-analysis

    bokeh serve --allow-websocket-origin=localhost:5006 --allow-websocket-origin=llruicms01:5006 --dev --args --hist-folder=/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v3/ -- bokeh_apps/cluster3D.py

Change the hist folder if needed (you should probably update the version number)

Then connect :
 - if on LLR network, then simply open <http://llruicms01:5006> in your browser (you may need to change the default port in case it is already in use, the port chosen is given by the previous command)
 - elsewhere, you need to create an SSH tunnel (you may also need to adapt the port): `ssh -NL 5006:llruicms01:5006 <your_username>@llrgate01.in2p3.fr`
   then open <http://localhost:5006> in your browser

## Code organization
### HistogramLib
Common code for bokeh and histogram projection, not specific to this analysis

### hists
Where all histograms are defined and loaded from pandas dataframe

### fill_histogram.py
main script to fill the histograms

### bokeh_apps
bokeh code common to all endpoints

### rechits.py, cluster2D.py, cluster3D.py
endpoints for bokeh server, meant to be run using `bokeh serve`