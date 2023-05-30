import uproot
import awkward as ak
import tqdm


class BaseComputation:
    neededBranches = list()
    """ List of branches needed for the computation """

    def process(self, array:ak.Array) -> None:
        pass


def computeAllFromTree(tree:uproot.TTree|list[uproot.TTree], computations:list[BaseComputation], count=None):
    neededBranches = set()
    for comp in computations:
        neededBranches.update(comp.neededBranches)
    

    if isinstance(tree, uproot.TTree):
        with tqdm.tqdm(total=tree.num_entries) as pbar:
            for (array, report) in tree.iterate(step_size="200MB", library="ak", report=True, filter_name=list(neededBranches)):
                for comp in computations:
                    comp.process(array)
                pbar.update(report.stop-report.start)
    else:
        for array in tqdm.tqdm(uproot.iterate(tree, step_size="200MB", library="ak", report=False, filter_name=list(neededBranches)),
                            total=count):
            for comp in computations:
                comp.process(array)


