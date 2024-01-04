

from collections import namedtuple
import dataclasses
from dataclasses import dataclass
import typing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import math
import functools
import itertools
import collections

import numpy as np
import scipy
import hist
import uncertainties
from uncertainties import unumpy
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm.auto import tqdm

from energy_resolution.fit import GaussianIterativeFitter, IterativeFitFailed, CruijffFitter, CruijffPdf
from hists.parameters import synchrotronBeamEnergiesMap

#NB : import zfit after importing energy_resolution.fit so we use the warnings filter defined there
import zfit

SigmaMuResult = namedtuple("SigmaMuResult", ["mu", "sigma", "fitResult", "fitQuality"])
""" Results of gaussian fit, as uncertainties ufloat. fitResult is the zfit FitResult. 
Attributes : 
 - mu, sigma : zfit Parameters
 - fitResult : zfit.minimizers.fitresult.FitResult
 - fitQuality : can be "good" or "bad" (bad can happen if some of steps of iterative fitting failed)
"""

class SigmaOverEFitError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def convert2DHistogramToDictOfHistograms(h_2D:hist.Hist) -> dict[int, hist.Hist]:
    """ Convert a 2D histogram with beamEnergy, X axises 
    to a dict of hists : { beamEnergy : h_2D[hist.loc(beamEnergy), :] }
    Removes empty slices
    """
    res_dict = dict()
    for beamEnergy in h_2D.axes["beamEnergy"]:
        sliced_h = h_2D[{"beamEnergy":hist.loc(beamEnergy)}]
        if sliced_h.sum() > 0:
            res_dict[beamEnergy] = sliced_h
    return res_dict

class SigmaOverEComputations:
    """ Class to manage gaussian fits of distribution of reconstructed energy """
    def __init__(self, sigmaWindow:tuple[float, float]=(1, 2.5), plotDebug=False, recoverFromFailedFits=False, fitFunction="truncatedGaussian") -> None:
        """ Parameters : 
         - sigmaWindow : window around the mean (as tuple (down, up)) for fitting a gaussian, in unit of sigma (nb down must be positive)
         - plotDebug : plot intermediate fit results (to be used in notebook)
         - recoverFromFailedFits : will try to recover as much as possible from failed fits. At least half of the fits must succeed
         - fitFunction : can be truncatedGaussian or cruijff 
        """
        self.sigmaWindow = sigmaWindow
        self.plotDebug = plotDebug
        self.recoverFromFailedFits = recoverFromFailedFits
        self.fitFunction = fitFunction


    def singleFit(self, rechit_energies_h:hist.Hist, fitter_kwargs=dict()) -> SigmaMuResult|None:
        """ Iterative fit of rechits_energies_h histogram with a gaussian
        Returns : a SigmaMuResult object.
        In case self.recoverFromFailedFits is True, can return an object with SigmaMuResult.fitQuality = "bad", or None in case no fit result could be obtained
        In case self.recoverFromFailedFits is False, will either return a good SigmaMuResult or raise IterativeFitFailed"""
        if self.fitFunction == "truncatedGaussian":
            fitter = GaussianIterativeFitter(rechit_energies_h, sigmaWindow=self.sigmaWindow)
            try:
                fitRes = fitter.multiIteration(plotDebug=self.plotDebug, **fitter_kwargs)
                fitQuality = "good"
            except IterativeFitFailed as e:
                if not self.recoverFromFailedFits:
                    raise
                if e.lastGoodFitResult is not None:
                    fitRes = e.lastGoodFitResult
                    fitQuality = "bad"
                else:
                    return None
            params = [fitter.params.mu, fitter.params.sigma]
            mu, sigma = uncertainties.correlated_values([fitRes.values[param] for param in params], 
                    fitRes.covariance(params))

        elif self.fitFunction == "cruijff":
            fitter = CruijffFitter(rechit_energies_h)
            try:
                fitRes = fitter.doFit()
                fitQuality = "good"
            except zfit.minimizers.strategy.FailMinimizeNaN:
                if not self.recoverFromFailedFits:
                    raise
                else:
                    return None
        
            params = [fitter.params.mu, fitter.params.sigma_L, fitter.params.sigma_R, fitter.params.alpha_L, fitter.params.alpha_R]
            values_uncert = uncertainties.correlated_values([fitRes.values[param] for param in params], 
                        fitRes.covariance(params))
            mu = values_uncert[0] # mu
            sigma = (values_uncert[1] + values_uncert[2]) / 2 # take mean of sigmaR and sigmaL

        else:
            raise ValueError("fitFunction should be truncatedGaussian or cruijff ")

        fitRes.freeze()  # freeze to make the FitResult pickleable
        return SigmaMuResult(mu, sigma, fitRes, fitQuality)

    def compute(self, h_per_energy:dict[int, hist.Hist], multiprocess=False, tqdm_dict=dict()) -> dict[int, SigmaMuResult]:
        """ Make all the gaussian fits for the histograms given
        Parameters : 
         - h_per_energy : dict beamEnergy -> histogram of reconstructed energy (or a 2D histogram with axes beamEnergy, recoEnergy)
         - multiprocess : if not False, use ProcessPoolExecutor for fitting all energies in parallel. (can be False, True (uses default start method), "fork", "forkserver", "spawn", a string being passed as a multiprocessing start method)
        Returns : a dictionnary beamEnergy -> SigmaMuResult, which is a namedtuple with mu and sigma as uncertainties package floats
        (includes correlation)
        """
        if isinstance(h_per_energy, hist.Hist):
            # convert 2D histogram to dict of histograms
            h_per_energy = convert2DHistogramToDictOfHistograms(h_per_energy)
        
        self.h_per_energy = h_per_energy
        if multiprocess is not False:
            if isinstance(multiprocess, str):
                ctx = multiprocessing.get_context(multiprocess)
            else:
                ctx = None
            with ProcessPoolExecutor(max_workers=10, mp_context=ctx) as executor:
                self.results = dict(zip(h_per_energy.keys(), executor.map(functools.partial(self.singleFit, fitter_kwargs=dict(progressBar=False)), h_per_energy.values())))
        else:
            self.results = dict(zip(h_per_energy.keys(), map(self.singleFit, tqdm(h_per_energy.values(), desc="Fitting", **tqdm_dict))))
        
        if self.recoverFromFailedFits:
            good_results = {beamEnergy : result for beamEnergy, result in self.results.items() if result is not None}
            if len(good_results)*2 < len(self.results) or len(good_results) < 3:
                raise SigmaOverEFitError(f"Not enough good fit results ({len(good_results)} fits succeeded out of {len(self.results)})")
            self.results = good_results

        return self.results
    
    def plotFitResult(self, beamEnergy:int, ax=None, sim=False, rebin=1):
        result = self.results[beamEnergy]
        mu, sigma = result.mu.nominal_value, result.sigma.nominal_value
        energy_h = self.h_per_energy[beamEnergy][::hist.rebin(rebin)]
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        fitWindow = mu - self.sigmaWindow[0]*sigma, mu + self.sigmaWindow[1]*sigma

        xlimits = (mu - 5*sigma, mu+4*sigma)
        ax.set_xlim(xlimits)
        
        if self.fitFunction == "truncatedGaussian":
            fitWindow_plot = ax.axvspan(*fitWindow, alpha=0.5, facecolor="0.5", label="Fit window")

        h_plot = hep.histplot(energy_h, flow="none", ax=ax, label="Energy distribution", histtype="errorbar", color="black")

        # Plotting pdf
        # Note : the normalization has to be handled carefully since we only fitted over a subset
        x_pdf = np.linspace(*xlimits, num=1000)
        if self.fitFunction == "truncatedGaussian":
            pdf = zfit.pdf.Gauss(mu, sigma, zfit.Space("x"))
            y_pdf = zfit.run(pdf.pdf(x_pdf, norm=fitWindow))
            # Scale with the nb of events in the fit, but then correct for bin width over whole plot
            scaleFactor = energy_h[hist.loc(fitWindow[0]):hist.loc(fitWindow[1])].sum() / energy_h.axes[0].size * (energy_h.axes[0].edges[-1]-energy_h.axes[0].edges[0])
            label = "Gaussian fit"
            fitResText = f"Fit results :\n$\\mu = {result.mu:L}$\n$\\sigma = {result.sigma:L}$"
        elif self.fitFunction == "cruijff":
            pdf = CruijffPdf(obs=zfit.Space("x"), **{key: val_dict["value"] for key, val_dict in result.fitResult.params.items()} )
            y_pdf = zfit.run(pdf.pdf(x_pdf, norm=xlimits))
            scaleFactor = energy_h[hist.loc(xlimits[0]):hist.loc(xlimits[1])].sum() / energy_h.axes[0].size * (energy_h.axes[0].edges[-1]-energy_h.axes[0].edges[0])
            label = "Cruijff fit"

            fitResText = f"Fit results :\n$\\sigma = {result.sigma:L}$\n"
            fitRes = result.fitResult
            for param in fitRes.params:
                param_uncert = uncertainties.ufloat(fitRes.params[param]['value'], math.sqrt(fitRes._covariance_dict['minuit_hesse'][param, param]))

                fitResText += f"$\\{param} = {param_uncert:L}$\n"
        
        
        pdf_plot,  = ax.plot(x_pdf, y_pdf*scaleFactor, label=label)
        
        if self.fitFunction == "truncatedGaussian":
            ax.legend(handles=[h_plot[0][0], pdf_plot, fitWindow_plot])
        else:
            ax.legend(handles=[h_plot[0][0], pdf_plot])

        #fitRes_plot, = ax.plot([], [], " ", label=f"Fit results :\nmean = {result.mu}\nsigma = {result.sigma}")
        ax.text(0.05, 0.8, s=fitResText,
                transform=ax.transAxes, fontsize=22, va="top")

        ax.set_ylabel("Event count")

        if sim:
            hep.cms.text("Simulation Preliminary")
        else:
            hep.cms.text("Preliminary")
        hep.cms.lumitext(f"{beamEnergy} GeV - $e^+$ TB")

        return fig



@dataclass
class EResolutionFitResult:
    """ Fit result of sigma over mean, for a quadratic sum of stochastic and constant terms. Values are uncertainties package floats (with correlations) """
    S:uncertainties.ufloat
    C:uncertainties.ufloat
    
    def __str__(self):
        return f"$S = ({self.S*100:L})" r"\sqrt{GeV} \%$" "\n" f"$C = ({self.C*100:L}) \%$"


def sigmaOverE_fitFunction(x, S, C):
    """ x is 1/sqrt(E), S and C are parameters """
    return np.sqrt((x*S)**2 + C**2)

class EnergyResolutionFitError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def fitSigmaOverE(sigmaOverEValues:dict[int, SigmaMuResult]) -> EResolutionFitResult:
    """ Fit the given sigma and mu values, as sigma/mu = quadratic_sum( S/sqrt(E_beam) ; C) 
    Returns EResolutionFitResult object
    Raises
     - RuntimeError in case of failed fit (scipy.optimize.OptimizeWarning in case covariance could not be estimated)
     - TypeError in case there less than 2 values for the fit
    """
    if len(sigmaOverEValues) < 2:
        raise SigmaOverEFitError("Not enough sigma/E points to fit : only energies " + " ".join((str(val) for val in sigmaOverEValues.keys())) + " were provided")
    xValues = [1/math.sqrt(synchrotronBeamEnergiesMap[beamEnergy]) for beamEnergy in sigmaOverEValues.keys()]
    # yValues are uncertainties float
    yValues = np.array([sigmaMuResult.sigma / sigmaMuResult.mu for sigmaMuResult in sigmaOverEValues.values()])
    
    (S, C), covMatrix = scipy.optimize.curve_fit(sigmaOverE_fitFunction, 
        xdata=xValues,
        ydata=unumpy.nominal_values(yValues), sigma=unumpy.std_devs(yValues),
        p0=[0.22, 0.6*1e-3], # starting values of parameters
        absolute_sigma=True, # units of sigma are the same as units of ydata, so this is appropriate
    )
    (S, C) = uncertainties.correlated_values([S, C], covMatrix)
    return EResolutionFitResult(S, C)


def fitSigmaOverEFromEnergyDistribution(h_per_energy:dict[int, hist.Hist]) -> EResolutionFitResult:
    """ Do the two steps : fit gaussian to all energies to get sigma and <E>, then fit sigma/<E> to get S and C """
    return fitSigmaOverE(SigmaOverEComputations().compute(h_per_energy))


# def fitSigmaOverEValues(sigmaOverEValues:dict[int, SigmaMuResult]):
#     x_space = zfit.Space("sigma_over_e", limits=(min(sigmaOverEValues.keys()), max(sigmaOverEValues.keys())))
#     data = zfit.Data.from_numpy(x_space, )

@dataclass(frozen=True)
class SigmaOverEPlotElement:
    """ Plot element of sigma/<E> """
    legend:str
    """ text for legend """
    fitResult:EResolutionFitResult = None
    """ EResolutionFitResult object (values of S and C) """
    fitFunction:typing.Callable[[float, float, float], float] = None
    """ Fit function of sigma over E (usually stochastic +(quadratic) constant terms)"""
    dataPoints:dict[int, uncertainties.ufloat] = None
    """ dict beamEnergy -> sigma/<E> value (as ufloat) """
    sigmaMuResults:dict[int, SigmaMuResult] = None
    """ dict beamEnergy -> SigmaMuResult (with mu and sigma values) """
    color:str = "blue"
    """ mpl color """
    legendGroup:str = None
    """ different plot elements with the same value of legendGroup will see their legend together """

def plotSigmaOverMean(plotElements:list[SigmaOverEPlotElement], ax:plt.Axes=None, xMode="E", 
        errors=True, plotFit=False, sim=False, linkPointsWithLines=True, markersize=5, legend=True, lumitext=None) -> matplotlib.figure.Figure:
    """ Make plots of sigma over <E> as a function of E or of 1/sqrt(E)
    Parameters : 
     - plotElements : list of plot elements
     - ax : a matplotlib Axes (if not specified creates a new figure)
     - xMode : can be "E" or ""1/sqrt(E)"
     - errors : if True plot errors on individual points
     - plotFit : if True, plot the fitted line as a dashed line
     - linkPointsWithLines : if True, link datapoints with broken lines
     - markersize : size of datapoints markers
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    
    if xMode == "E":
        ax.set_xlabel("Incident beam energy (GeV)")
    elif xMode == "1/sqrt(E)":
        ax.set_xlabel(r"$\frac{1}{\sqrt{E_{i}^{incident} [GeV]}}$")
    else:
        raise ValueError("xMode must be E or 1/sqrt(E)")

    handles = collections.defaultdict(list)
    for plotElement in plotElements:
        cur_handle_group = handles[plotElement.legendGroup]
        if plotElement.dataPoints is not None:
            xValues = np.array([synchrotronBeamEnergiesMap[beamEnergy] for beamEnergy in plotElement.dataPoints.keys()])
            if xMode == "1/sqrt(E)":
                xValues = 1/np.sqrt(xValues)
            
            yValues = np.fromiter(plotElement.dataPoints.values(), dtype=object)
            yValues_nominal = unumpy.nominal_values(yValues)

            common_kwargs = dict(marker='o', markersize=markersize, fillstyle="full", label=plotElement.legend, color=plotElement.color)
            if not linkPointsWithLines:
                common_kwargs["linestyle"] = "none"
            
            if errors:
                h = ax.errorbar(xValues, yValues_nominal, yerr=unumpy.std_devs(yValues),  
                        capsize=1.5, elinewidth=1.5,**common_kwargs)
            else:
                h = ax.plot(xValues, yValues_nominal, **common_kwargs)[0]

            cur_handle_group.append(h)
        else:
            assert False, "Must supply data points"
        
        if plotElement.fitResult is not None:
            linestyle = '--' if plotFit else "None"
            if plotElement.dataPoints is None:
                label = plotElement.legend + "\n"
            else:
                label = ""
            label += str(plotElement.fitResult)
        
            fitFct_params_dict = {key : value.nominal_value for key, value in dataclasses.asdict(plotElement.fitResult).items()}
            if xMode == "1/sqrt(E)":
                fitFctBound = functools.partial(plotElement.fitFunction, **fitFct_params_dict)
                h = ax.axline((xValues[0], fitFctBound(xValues[0])), (xValues[-1], fitFctBound(xValues[-1])),
                    linestyle=linestyle, color=plotElement.color,
                    label=label)
            elif xMode == "E":
                fitFctBound = lambda E : plotElement.fitFunction(1/math.sqrt(E), **fitFct_params_dict)
                xVals_function = np.linspace(xValues[0], xValues[-1], 500)
                h, = ax.plot(xVals_function, [fitFctBound(x) for x in xVals_function], linestyle=linestyle, color=plotElement.color,
                    label=label)
            
            cur_handle_group.append(h)


    ax.set_ylabel(r"$\frac{\hat{\sigma}_i}{\hat{\mu}_i}$")
    # if sim:
    #     hep.cms.text("Simulation Preliminary", ax=ax)
    # else:
    #     hep.cms.text("Preliminary", ax=ax)
    hep.cms.lumitext(lumitext if lumitext else f"$e^+$ Test Beam", ax=ax)
    
    if not legend:
        return fig
    legend_kwargs = dict(frameon=True, handletextpad=0.3)
    if xMode == "E":
        legend_positions = ["upper right", "lower left"]
    elif xMode == "1/sqrt(E)":
        legend_positions = ["upper left", "lower right"]
    for handle_group, loc in zip(handles.values(), itertools.chain(legend_positions, itertools.repeat(None))):
        ax.add_artist(ax.legend(handles=handle_group, loc=loc, **legend_kwargs))

    return fig



# Taken from https://github.com/botprof/plotting-uncertainty-ellipses/blob/main/plotting-uncertainty-ellipses.ipynb
def plotEllipse(x:uncertainties.ufloat, y:uncertainties.ufloat, cl:float=0.95, ellipse_kwargs:dict=dict()) -> matplotlib.patches.Ellipse:
    """ Builds a matplotlib.patches.ellipse depicting the confidence area for the two variables x and y """
    cov = uncertainties.covariance_matrix([x, y])
    W, V = np.linalg.eig(cov)
    j_max = np.argmax(W)
    j_min = np.argmin(W)
    s = scipy.stats.chi2.isf(1-cl, 2) # 1-confidence level, 2 degrees of freedom
    return matplotlib.patches.Ellipse(
        (x.nominal_value, y.nominal_value),
        2.0 * np.sqrt(s * W[j_max]),
        2.0 * np.sqrt(s * W[j_min]),
        angle=np.arctan2(V[j_max, 1], V[j_max, 0]) * 180 / np.pi,
        **ellipse_kwargs
    )

def plotSCAsEllipse(plotElements:list[SigmaOverEPlotElement], detailedLabels=False) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots()
    
    for plotElement in plotElements:
        S_scaled, C_scaled = plotElement.fitResult.S*100, plotElement.fitResult.C*100
        x, y = S_scaled.nominal_value, C_scaled.nominal_value
        ax.plot([x], [y], ".", label=plotElement.legend, color=plotElement.color)
        ax.add_patch(plotEllipse(S_scaled, C_scaled, ellipse_kwargs=dict(color=plotElement.color, alpha=0.5)))
        ax.annotate(plotElement.legend, xy=(x, y))

    if detailedLabels:
        ax.set_xlabel(r"Stochastic term $S (\sqrt{GeV} \%)$")
        ax.set_ylabel(r"Constant term C (%)")
    else:
        ax.set_xlabel(r"$S (\sqrt{GeV} \%)$")
        ax.set_ylabel(r"C (%)")
    ax.legend()

    return fig

def plotFittedMean(datatype_dict:dict[str, dict[int, SigmaMuResult]]|list[SigmaOverEPlotElement], ax:plt.Axes=None, normByRechits=False, errors=False,
            levelLabelMap:dict[str, str]=dict(), beamEnergiesToCircle:dict[str, list[int]]|bool=None):
    """ Plot fitted mu as a function of beam energy
    Parameters : 
     - datatype_dict : nested dicts :  level -> beamEnergy ->  SigmaMuResult
     - levelLabelMap : dict mapping level (key of datatype_dict) -> legend label
     - beamEnergiesToCircle : dict : level -> list of beam energy values to circle (meant for circling test points in ML)
            can also be set to True, in which case it will look into the SigmaOverEPlotElement given
    """
    if isinstance(datatype_dict, list):
        # convert SigmaOverEPlotElement to dict
        datatype_dict = {plotElt.legend : plotElt.sigmaMuResults for plotElt in datatype_dict}
        if beamEnergiesToCircle is True:
            beamEnergiesToCircle = {}
            for plotElt in datatype_dict:
                try:
                    beamEnergiesToCircle[plotElt.legend] = plotElt.beamEnergiesInTestSet
                except:
                    pass
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if normByRechits:
        datatype_dict = datatype_dict.copy()
        rechits_dict = datatype_dict.pop("rechits")
        
        plot_dict = {level : np.array([level_dict[beamEnergy].mu / rechits_dict[beamEnergy].mu for beamEnergy in level_dict]) 
            for level, level_dict in datatype_dict.items()}
        ax.set_ylabel("$\\frac{<E>}{<E_{rechits}>}$")

    else:
        plot_dict = {level : np.array([level_dict[beamEnergy].mu / synchrotronBeamEnergiesMap[beamEnergy] for beamEnergy in level_dict]) 
            for level, level_dict in datatype_dict.items()}
        ax.set_ylabel("$\\frac{\hat{\mu}_i}{E_{i}^{incident}}$")
    
    for level, plot_level_values in plot_dict.items():
        xValues = [synchrotronBeamEnergiesMap[beamEnergy] for beamEnergy in datatype_dict[level]]
        yValues = unumpy.nominal_values(plot_level_values)
        if errors:
            ax.errorbar(xValues, yValues, yerr=unumpy.std_devs(plot_level_values),
                fmt='-', label=levelLabelMap.get(level, level), capsize=1.5, ecolor="black", elinewidth=1.5, barsabove=True)
        else:
            ax.plot(xValues, yValues, 'o-', label=levelLabelMap.get(level, level))

        if beamEnergiesToCircle is not None and level in beamEnergiesToCircle:
            allBeamEnergies = list(datatype_dict[level].keys())
            xToCircle = []
            yToCircle = []
            for beamEnergy in beamEnergiesToCircle[level]:
                idx = allBeamEnergies.index(beamEnergy)
                xToCircle.append(xValues[idx])
                yToCircle.append(yValues[idx])
            ax.scatter(xToCircle, yToCircle, s=500, facecolors="none", edgecolors="red", label="Test points")

    ax.set_xlabel("Incident beam energy (GeV)")
    
    ax.axhline(y=1, linestyle="--", color="black")
    #hep.cms.text("Preliminary", ax=ax)
    hep.cms.lumitext(f"$e^+$ Test Beam", ax=ax)
    ax.legend()

    return fig

def plotFittedSigma(datatype_dict:dict[str, dict[int, SigmaMuResult]]|list[SigmaOverEPlotElement], ax:plt.Axes=None, normBy:str=None, errors=False,
            levelLabelMap:dict[str, str]=dict(), beamEnergiesToCircle:dict[str, list[int]]|bool=None):
    """ Plot fitted mu as a function of beam energy
    Parameters : 
     - datatype_dict : nested dicts :  level -> beamEnergy ->  SigmaMuResult
     - normBy : can be None (no normalization), "rechits" (norm to rechits)
     - levelLabelMap : dict mapping level (key of datatype_dict) -> legend label
     - beamEnergiesToCircle : dict : level -> list of beam energy values to circle (meant for circling test points in ML)
            can also be set to True, in which case it will look into the SigmaOverEPlotElement given
    """
    if isinstance(datatype_dict, list):
        # convert SigmaOverEPlotElement to dict
        datatype_dict = {plotElt.legend : plotElt.sigmaMuResults for plotElt in datatype_dict}
        if beamEnergiesToCircle is True:
            beamEnergiesToCircle = {}
            for plotElt in datatype_dict:
                try:
                    beamEnergiesToCircle[plotElt.legend] = plotElt.beamEnergiesInTestSet
                except:
                    pass
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if normBy == "rechits":
        datatype_dict = datatype_dict.copy()
        rechits_dict = datatype_dict.pop("rechits")
        
        plot_dict = {level : np.array([level_dict[beamEnergy].sigma / rechits_dict[beamEnergy].sigma for beamEnergy in level_dict]) 
            for level, level_dict in datatype_dict.items()}
        ax.set_ylabel("$\\frac{\hat{\sigma}_i}{\hat{\sigma}_{rechits}}$")
        ax.axhline(y=1, linestyle="--", color="black")

    elif normBy == "sqrt(E)":
        plot_dict = {level : np.array([level_dict[beamEnergy].sigma / math.sqrt(synchrotronBeamEnergiesMap[beamEnergy]) for beamEnergy in level_dict]) 
            for level, level_dict in datatype_dict.items()}
        ax.set_ylabel(r"$\frac{\hat{\sigma}_i}{\sqrt{E_i^{incident}}}$")
    elif normBy is None:
        plot_dict = {level : np.array([level_dict[beamEnergy].sigma for beamEnergy in level_dict]) 
            for level, level_dict in datatype_dict.items()}
        ax.set_ylabel(r"$\hat{\sigma}_i$ (GeV)")
    else:
        raise ValueError(f"Invalid normBy={normBy}")

    for level, plot_level_values in plot_dict.items():
        xValues = [synchrotronBeamEnergiesMap[beamEnergy] for beamEnergy in datatype_dict[level]]
        yValues = unumpy.nominal_values(plot_level_values)
        if errors:
            ax.errorbar(xValues, yValues, yerr=unumpy.std_devs(plot_level_values),
                fmt='-', label=levelLabelMap.get(level, level), capsize=1.5, ecolor="black", elinewidth=1.5, barsabove=True)
        else:
            ax.plot(xValues, yValues, 'o-', label=levelLabelMap.get(level, level))

        if beamEnergiesToCircle is not None and level in beamEnergiesToCircle:
            allBeamEnergies = list(datatype_dict[level].keys())
            xToCircle = []
            yToCircle = []
            for beamEnergy in beamEnergiesToCircle[level]:
                idx = allBeamEnergies.index(beamEnergy)
                xToCircle.append(xValues[idx])
                yToCircle.append(yValues[idx])
            ax.scatter(xToCircle, yToCircle, s=500, facecolors="none", edgecolors="red", label="Test points")

    ax.set_xlabel("Incident beam energy (GeV)")
        
    hep.cms.text("Preliminary", ax=ax)
    hep.cms.lumitext(f"$e^+$ Test Beam", ax=ax)
    ax.legend()

    return fig