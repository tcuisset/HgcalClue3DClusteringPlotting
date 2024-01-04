import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import hist
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import zfit
import numpy as np

from hists.parameters import beamEnergies
from energy_resolution.hist_loader import HistLoader
from energy_resolution.sigma_over_e import SigmaOverEComputations
from energy_resolution.fit import HistogramEstimates
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader

# Code taken from zfit docs (for convenience in notebooks)
def get_param(name, value=None, lower=None, upper=None, step_size=None, **kwargs):
    """Either create a parameter or return existing if a parameter with this name already exists.

    If anything else than *name* is given, this will be used to change the existing parameter.

    Args:
        name: Name of the Parameter
        value : starting value
        lower : lower limit
        upper : upper limit
        step_size : step size

    Returns:
        ``zfit.Parameter``
    """
    all_params = zfit.core.parameter.ZfitParameterMixin._existing_params
    if name in all_params:
        parameter = all_params[name]
        if lower is not None:
            parameter.lower = lower
        if upper is not None:
            parameter.upper = upper
        if step_size is not None:
            parameter.step_size = step_size
        if value is not None:
            parameter.set_value(value)
        return parameter

    # otherwise create new one
    parameter = zfit.Parameter(name, value, lower, upper, step_size)
    #all_params[name] = parameter
    return parameter


def doFit(mean):
    obs = zfit.Space('x', limits=(-10, 10))
    data_np = np.random.normal(mean, 1, size=10000)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    mu = get_param("mu", 2.4, -1, 5)
    sigma = get_param("sigma", 1.3,  0, 5)


    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    # Stage 1: create an unbinned likelihood with the given PDF and dataset
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit minimizer)
    minimizer = zfit.minimize.Minuit()

    # Stage 3: minimise the given negative likelihood
    result = minimizer.minimize(nll)

    return mu.value()

if __name__ == "__main__":

    #with ProcessPoolExecutor(max_workers=2) as executor:
    #    for res in executor.map(doFit, [0, 1, 2, 3]):
    #        print(res)
        #self.results = dict(zip(h_per_energy.keys(), executor.map(self.singleFit, h_per_energy.values())))

    reader = ClueNtupleReader("v40", "cmssw", "data")
    loader = HistLoader(reader.histStore)

    comp_sigma_e = SigmaOverEComputations(sigmaWindow=(1, 2.5))
    datatype = "data"

    sigma_e_results = comp_sigma_e.compute({beamEnergy : loader.getRechitsProjected(datatype, beamEnergy) for beamEnergy in beamEnergies}, multiprocess=True)

    print(sigma_e_results)