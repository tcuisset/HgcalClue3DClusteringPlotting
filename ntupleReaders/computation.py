import uproot
import awkward as ak
from tqdm.auto import tqdm


class BaseComputation:
    neededBranches = list()
    """ List of branches needed for the computation """

    def process(self, array:ak.Array) -> None:
        pass


def computeAllFromTree(tree:uproot.TTree|list[uproot.TTree], computations:list[BaseComputation], tqdm_options:dict=dict()):
    """ Process all given Computation objects from all events in tree 
    Parameters : 
     - tree : an uproot TTree to process, or a list of them
     - computations : a list of computations (they must satifsfy the BaseComputation model)
     - tqdm_options : dict of keyword args passed to tqdm. You can pass for example desc, or count (in case you provide an generator for tree which has no __len__)
     """
    neededBranches = set()
    for comp in computations:
        neededBranches.update(comp.neededBranches)
    

    if isinstance(tree, uproot.TTree):
        with tqdm(total=tree.num_entries, **tqdm_options) as pbar:
            for (array, report) in tree.iterate(step_size="200MB", library="ak", report=True, filter_name=list(neededBranches)):
                for comp in computations:
                    comp.process(array)
                pbar.update(report.stop-report.start)
    else:
        for array in tqdm.tqdm(uproot.iterate(tree, step_size="200MB", library="ak", report=False, filter_name=list(neededBranches)),
                            **tqdm_options):
            for comp in computations:
                comp.process(array)


