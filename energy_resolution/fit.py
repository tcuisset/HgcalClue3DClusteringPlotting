import math
import typing
from functools import cached_property

# Note : zfit uses tensorflow, which by default will use all the memory of the first GPU if there is one
# So disable the use of GPUs by tensorflow (must be done before any zfit object is created)
# it will only use the CPU
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import zfit

import numpy as np
import hist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mplhep as hep

from hists.parameters import beamEnergies

class HistogramEstimates:
    def __init__(self, h:hist.Hist) -> None:
        self.h = h
        if h.sum() == 0:
            raise ValueError("Histogram is empty")
    
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
def get_param(name, value, lower=None, upper=None, step_size=None, **kwargs):
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
        parameter.lower = lower
        parameter.upper = upper
        parameter.step_size = step_size
        parameter.set_value(value)
        return parameter

    # otherwise create new one
    parameter = zfit.Parameter(name, value, lower, upper, step_size)
    #all_params[name] = parameter
    return parameter

class GaussianParameters:
    def __init__(self) -> None:
        self.mu = get_param("mu", 100)
        self.sigma = get_param("sigma", 1.)

    def adaptParametersToHist(self, h:hist.Hist):
        estimates = HistogramEstimates(h)
        # Be robust with outliers
        mean_h, sigma_h = estimates.median, estimates.sigmaEstimateUsingIQR
        self.mu.lower_limit, self.mu.upper_limit = max(0, mean_h*0.5), max(350, mean_h*1.5)
        self.mu.assign(mean_h)

        self.sigma.lower_limit, self.sigma.upper_limit = 0.1*sigma_h, 5*sigma_h
        self.sigma.assign(sigma_h)
    
    def print(self):
        print(self.mu)
        print(self.sigma)
        

class SingleFitter:
    def __init__(self, h:hist.Hist, params:GaussianParameters=None) -> None:
        if params is None:
            params = GaussianParameters()
        
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
    
    hep.histplot(h, ax=ax, histtype="errorbar", flow="none")
    x_plot = np.linspace(h.axes[0].edges[0], h.axes[0].edges[0-1], num=1000)
    y_plot = zfit.run(pdf.pdf(x_plot, norm=space))
    ax.plot(x_plot, y_plot * h.sum() / h.axes[0].size * space.area())
    if text is not None:
        ax.text(0.15, 0.85, text, transform=ax.transAxes)


class IterativeFitFailed(Exception):
    def __init__(self, lastGoodFitResult:zfit.minimizers.fitresult.FitResult=None, successfulFitIterationsCount=0) -> None:
        super().__init__()
        self.lastGoodFitResult = lastGoodFitResult
        self.successfulFitIterationsCount = successfulFitIterationsCount


class GaussianIterativeFitter:
    def __init__(self, h:hist.Hist, sigmaWindow:tuple[float, float]) -> None:
        self.h = h
        self.sigmaWindow = sigmaWindow
        self.params = GaussianParameters()

        self.params.adaptParametersToHist(h)

    def fitIteration(self, plotDebugBeforeFit=False) -> tuple[zfit.minimizers.fitresult.FitResult, SingleFitter]:
        """ Single iteration of iterative fitting 
        Raises :
          - ValueError : in case the window is smaller than the binning of the histogram
          - zfit.minimizers.strategy.FailMinimizeNaN : when the fit failed
        """
        low_bound = self.params.mu.value().numpy() - self.sigmaWindow[0]*self.params.sigma.value().numpy()
        high_bound = self.params.mu.value().numpy() + self.sigmaWindow[1]*self.params.sigma.value().numpy()
        low_bound = max(self.h.axes[0].edges[0], low_bound) # If low_bound is negative then we get ValueError : begin < end required
        high_bound = min(self.h.axes[0].edges[-1], high_bound)
        h_windowed = self.h[hist.loc(low_bound):hist.loc(high_bound)]
        
        fitter = SingleFitter(h_windowed, self.params)
        if plotDebugBeforeFit:
            plot(fitter.h, fitter.unbinned_pdf, fitter.data.space, text="Before fit")
        return fitter.doFit(), fitter
    
    def multiIteration(self, maxIter=5, verbose=False, plotDebug=True, progressBar=True) -> zfit.minimizers.fitresult.FitResult:
        for i in tqdm(range(maxIter), desc=f"Iterative fitting - {self.params.mu.value().numpy():.0f} GeV", leave=None, disable=(not progressBar)):
            if verbose:
                self.params.print()
            try:
                fitResult, fitter = self.fitIteration(plotDebugBeforeFit=((i==0) and plotDebug))
            except (zfit.minimizers.strategy.FailMinimizeNaN, ValueError): # ValueError : begin < end required -> the window is too narrow
                raise IterativeFitFailed(lastGoodFitResult=fitResult if i > 0 else None, successfulFitIterationsCount=i)
            
            if plotDebug:
                plot(fitter.h, fitter.unbinned_pdf, fitter.data.space, text=f"After fit, iteration {i}")

        return fitResult
    