from typing import Any, Callable, Iterable, NamedTuple, Union

import numpy as np
import uncertainties
import uncertainties.umath
import hist
from hist import loc
from hist.plot import _fit_callable_to_hist, _construct_gaussian_callable

def iterative_gaussian_fit(h:hist.Hist,  maxIterations=10, startWindow=None, sigmaRange=(-2, 2), minSigma=None) -> tuple[tuple[float, float], str]:
    """ Do an iterative gaussian fit on an histogram
    Parameters : 
     - sigmaRange : target range for fit, in numbers of sigma down/up for ex (-3, 2) is [mean-3*sigma, mean+2*sigma]
    Returns a tuple
     - tuple (windowLow, windowHigh)
     - errorMode : can be hesse, curve_fit
    """
    assert(len(h.axes) == 1)
    
    if startWindow is None:
        window = h.axes[0].edges[0], h.axes[0].edges[-1]
    else:
        window = startWindow
    
    norm, mean, sigma = None, None, None
    iter_count = 0
    while iter_count < maxIterations:
        hist_view = h[loc(window[0]):loc(window[1])]

        if norm is None:
            gaussian = _construct_gaussian_callable(hist_view)
        else:
            # gauss is a closure that will get evaluated in _fit_callable_to_hist
            def gauss(
                x: np.typing.NDArray[Any],
                constant: float = norm,
                mean: float = mean,
                sigma: float = sigma,
            ) -> np.typing.NDArray[Any]:
                # Note: Force np.typing.NDArray[Any] type as numpy ufuncs have type "Any"
                ret: np.typing.NDArray[Any] = constant * np.exp(
                    -np.square(x - mean) / (2 * np.square(sigma))
                )
                return ret
            gaussian = gauss
            
        try:
            (
                _,_,_,
                bestfit_result
            )  = _fit_callable_to_hist(gaussian, hist_view, likelihood=True)
            errorMode = "hesse"
        except:
            try:
                (
                    _,_,_,
                    bestfit_result
                )  = _fit_callable_to_hist(gaussian, hist_view, likelihood=False)
                errorMode = "curve_fit"
            except:
                hist_view.plot()
                raise RuntimeError("Fit failure", window, (norm, mean, sigma))
        
        ((norm, mean, sigma), covarianceMatrix) = bestfit_result

        absSigma = abs(sigma) # For some reason negative sigma happens
        if absSigma < minSigma:
            absSigma = minSigma
            errorMode = errorMode + "+sigmaAtBoundary"

        window = mean + sigmaRange[0] * absSigma, mean + sigmaRange[1] * absSigma

        iter_count += 1
    try:
        param_values = list(uncertainties.correlated_values([norm, mean, sigma], covarianceMatrix))
    except:
        param_values = [uncertainties.ufloat([norm, mean, sigma][i], covarianceMatrix[i, i]) for i in range(3)]
    param_values[2] = abs(param_values[2]) # sigma
    return window, tuple(param_values), errorMode
