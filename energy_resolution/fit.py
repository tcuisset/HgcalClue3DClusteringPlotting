import math
import typing
from functools import cached_property

import numpy as np
import zfit
import hist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mplhep as hep

from hists.parameters import beamEnergies

class HistogramEstimates:
    def __init__(self, h:hist.Hist) -> None:
        self.h = h
    
    @cached_property
    def quantiles(self):
        # Code inspired from https://stackoverflow.com/a/29677616
        # Compute weighted quantiles
        weighted_quantiles_values = (np.cumsum(self.h.view()) - 0.5 * self.h.view()) / self.h.sum()

        quantiles = [0.25, 0.5, 0.75]
        quantiles_values = np.interp(quantiles, weighted_quantiles_values, self.h.axes[0].centers)
        return dict(zip(quantiles, quantiles_values))
    
    @property
    def mean(self):
        if len(self.h.axes) > 1:
                raise ValueError("Only 1D histogram supported")
        return np.average(self.h.axes[0].centers, weights=self.h.view(flow=False))
    
    @property
    def median(self):
        return self.quantiles[0.5]

    @property
    def sigma(self):
        return math.sqrt(np.average((self.h.axes[0].centers - self.mean)**2, weights=self.h.view(flow=False)))

    @property
    def sigmaEstimateUsingIQR(self):
        """ Compute an estimate of the standard deviation of the histogram using the interquartile range. 
        It is more robust to outliers than just taking the histogram standard deviation
        """
        iqr = self.quantiles[0.75] - self.quantiles[0.25] # interquantile range : range between 25% and 75% quantiles
        return iqr / 1.349 # Gaussian has an IQR of 1.349

# Code taken from zfit docs (for convenience in notebooks)
all_params = {}
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

class GaussianParameters:
    mu = get_param("mu", 100, 0., max(beamEnergies)*2)
    sigma = get_param("sigma", 1., 0., max(beamEnergies))

    @classmethod
    def adaptParametersToHist(cls, h:hist.Hist):
        estimates = HistogramEstimates(h)
        # Be robust with outliers
        mean_h, sigma_h = estimates.median, estimates.sigmaEstimateUsingIQR
        cls.mu.lower_limit, cls.mu.upper_limit = max(0, mean_h*0.5), max(350, mean_h*1.5)
        cls.mu.assign(mean_h)

        cls.sigma.lower_limit, cls.sigma.upper_limit = 0.1*sigma_h, 5*sigma_h
        cls.sigma.assign(sigma_h)
    
    @classmethod
    def print(cls):
        print(cls.mu)
        print(cls.sigma)
        

class SingleFitter:
    def __init__(self, h:hist.Hist, params:typing.Type[GaussianParameters]=GaussianParameters) -> None:
        self.h = h
        self.params = params
        self.data = zfit.data.BinnedData.from_hist(self.h)
        self.unbinned_pdf = zfit.pdf.Gauss(obs=self.data.obs, mu=self.params.mu, sigma=self.params.sigma)
        self.binned_pdf = zfit.pdf.BinnedFromUnbinnedPDF(self.unbinned_pdf, space=self.data.space)
    
    def doFit(self) -> zfit.minimizers.fitresult.FitResult:
        loss = zfit.loss.BinnedNLL(self.binned_pdf, self.data)
        minimizer = zfit.minimize.Minuit(verbosity=5)
        return minimizer.minimize(loss)


def plot(h:hist.Hist, pdf:zfit.pdf.BasePDF, space:zfit.Space, ax=None, text=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    hep.histplot(h, ax=ax, histtype="errorbar")
    x_plot = np.linspace(h.axes[0].edges[0], h.axes[0].edges[0-1], num=1000)
    y_plot = zfit.run(pdf.pdf(x_plot, norm=space))
    ax.plot(x_plot, y_plot * h.sum() / h.axes[0].size * space.area())
    if text is not None:
        ax.text(0.15, 0.85, text, transform=ax.transAxes)

class GaussianIterativeFitter:
    def __init__(self, h:hist.Hist, sigmaWindow:tuple[float, float]) -> None:
        self.h = h
        self.sigmaWindow = sigmaWindow
        self.params = GaussianParameters

        self.params.adaptParametersToHist(h)

    def fitIteration(self, plotDebugBeforeFit=False) -> tuple[zfit.minimizers.fitresult.FitResult, SingleFitter]:
        low_bound = self.params.mu.value().numpy() - self.sigmaWindow[0]*self.params.sigma.value().numpy()
        high_bound = self.params.mu.value().numpy() + self.sigmaWindow[1]*self.params.sigma.value().numpy()
        low_bound = max(self.h.axes[0].edges[0], low_bound) # If low_bound is negative then we get ValueError : begin < end required
        high_bound = min(self.h.axes[0].edges[-1], high_bound)
        h_windowed = self.h[hist.loc(low_bound):hist.loc(high_bound)]
        
        fitter = SingleFitter(h_windowed, self.params)
        if plotDebugBeforeFit:
            plot(fitter.h, fitter.unbinned_pdf, fitter.data.space, text="Before fit")
        return fitter.doFit(), fitter
    
    def multiIteration(self, maxIter=5, verbose=False, plotDebug=True) -> zfit.minimizers.fitresult.FitResult:
        for i in tqdm(range(maxIter), desc=f"Iterative fitting - {self.params.mu.value().numpy():.0f} GeV", leave=None):
            if verbose:
                GaussianParameters.print()
            fitResult, fitter = self.fitIteration(plotDebugBeforeFit=((i==0) and plotDebug))
            if plotDebug:
                plot(fitter.h, fitter.unbinned_pdf, fitter.data.space, text=f"After fit, iteration {i}")

        return fitResult
    