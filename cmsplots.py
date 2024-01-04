import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib import rcParams

import hist
import numpy as np

import mplhep as hep
from mplhep.label import ExpText, ExpSuffix

hep.label.exp_text
def cms_exp_text(
    text="",
    exp="CMS",
    loc=0,
    *,
    ax=None,
    fontname=None,
    fontsize=None,
    exp_weight="bold",
    italic=(False, True),
    pad=0,
):
    """Add typical LHC experiment primary label to the axes.
    Parameters
    ----------
        text : string, optional
            Secondary experiment label, typically not-bold and smaller
            font-size. For example "Simulation" or "Preliminary"
        loc : int, optional
            Label position:
            - 0 : Above axes, left aligned
            - 1 : Top left corner
            - 2 : Top left corner, multiline
            - 3 : Split EXP above axes, rest of label in top left corner"
            - 4 : Top right corner (not supported yet)
            - 5 : Top right corner, multiline
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, last one is fetched)
        fontname : string, optional
            Name of font to be used.
        fontsize : string, optional
            Defines size of "secondary label". Experiment label is 1.3x larger.
        exp_weight : string, optional
            Set fontweight of <exp> label. Default "bold".
        italic : (bool, bool), optional
            Tuple of bools to switch which label is italicized
        pad : float, optional
            Additional padding from axes border in units of axes fraction size.
    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`
            object
    """

    _font_size = rcParams["font.size"] if fontsize is None else fontsize
    fontname = "TeX Gyre Heros" if fontname is None else fontname

    if ax is None:
        ax = plt.gca()

    loc1_dict = {
        0: {"xy": (0.001, 1 + pad), "va": "bottom"},
        1: {"xy": (0.05, 0.95 - pad), "va": "top"},
        2: {"xy": (0.95 - pad, 0.95 - pad), "va" : "top", "ha": "right"},
    }

    loc2_dict = {
        0: {"xy": (0.001, 1.005 + pad), "va": "bottom"},
        1: {"xy": (0.05, 0.9550 - pad), "va": "bottom"},
        2: {"xy": (0.05, 0.9450 - pad), "va": "top"},
        3: {"xy": (0.05, 0.95 - pad), "va": "top"},
        4: {"xy": (0.95, 0.9550 - pad), "va": "top", "ha": "right"},
        5: {"xy": (0.95, 0.9450 - pad), "va": "top", "ha": "right"},
        6: {"xy": (0.05, 0.9550 - pad), "va": "bottom"},
    }

    if loc not in [0, 1, 2, 3, 5]:
        raise ValueError(
            "loc must be in {0, 1, 2}:\n"
            "0 : Above axes, left aligned\n"
            "1 : Top left corner\n"
            "2 : Top left corner, multiline\n"
            "3 : Split EXP above axes, rest of label in top left corner\n"
            "4 : Top right corner (not supported yet)\n"
            "5 : Top right corner, multiline\n"
        )

    def pixel_to_axis(extent, ax=None):
        # Transform pixel bbox extends to axis fractions
        if ax is None:
            ax = plt.gca()

        extent = extent.transformed(ax.transData.inverted())

        def dist(tup):
            return abs(tup[1] - tup[0])

        dimx, dimy = dist(ax.get_xlim()), dist(ax.get_ylim())
        x, y = ax.get_xlim()[0], ax.get_ylim()[0]
        x0, y0, x1, y1 = extent.extents

        return extent.from_extents(
            abs(x0 - x) / dimx,
            abs(y0 - y) / dimy,
            abs(x1 - x) / dimx,
            abs(y1 - y) / dimy,
        )

    if loc in [0, 3]:
        _exp_loc = 0
    elif loc in [1, 2]:
        _exp_loc = 1
    else:
        _exp_loc = 2
    _formater = ax.get_yaxis().get_major_formatter()
    if type(mpl.ticker.ScalarFormatter()) == type(_formater) and _exp_loc == 0:
        _sci_box = pixel_to_axis(
            ax.get_yaxis().offsetText.get_window_extent(ax.figure.canvas.get_renderer())
        )
        _sci_offset = _sci_box.width * 1.1
        loc1_dict[_exp_loc]["xy"] = (_sci_offset, loc1_dict[_exp_loc]["xy"][-1])
        if loc == 0:
            loc2_dict[_exp_loc]["xy"] = (_sci_offset, loc2_dict[_exp_loc]["xy"][-1])

    exptext = ExpText(
        *loc1_dict[_exp_loc]["xy"],
        text=exp,
        transform=ax.transAxes,
        ha=loc1_dict[_exp_loc].get("ha", "left"),
        va=loc1_dict[_exp_loc]["va"],
        fontsize=_font_size * 1.3,
        fontweight=exp_weight,
        fontstyle="italic" if italic[0] else "normal",
        fontname=fontname,
    )
    ax._add_text(exptext)

    _dpi = ax.figure.dpi
    _exp_xoffset = (
        exptext.get_window_extent(ax.figure.canvas.get_renderer()).width / _dpi * 1.05
    )
    if loc == 0:
        _t = mtransforms.offset_copy(
            exptext._transform, x=_exp_xoffset, units="inches", fig=ax.figure
        )
    elif loc in [1]:
        _t = mtransforms.offset_copy(
            exptext._transform,
            x=_exp_xoffset,
            y=-exptext.get_window_extent().height / _dpi,
            units="inches",
            fig=ax.figure,
        )
    elif loc in [2, 5]:
        _t = mtransforms.offset_copy(
            exptext._transform,
            y=-exptext.get_window_extent().height / _dpi,
            units="inches",
            fig=ax.figure,
        )
    elif loc == 3:
        _t = mtransforms.offset_copy(exptext._transform, units="inches", fig=ax.figure)

    expsuffix = ExpSuffix(
        *loc2_dict[loc]["xy"],
        text=text,
        transform=_t,
        ha=loc2_dict[loc].get("ha", "left"),
        va=loc2_dict[loc]["va"],
        fontsize=_font_size,
        fontname=fontname,
        fontstyle="italic" if italic[1] else "normal",
    )
    ax._add_text(expsuffix)

    return exptext, expsuffix


def plotFillBetweenHistogram(h:hist.Hist|tuple, ax, shift=False, confidenceIntervals=None, **kwargs):
    """ Plot a histogram using plt.fillbetween, useful for MC uncertainties 
    Parameters : 
     - h : the histogram to plot central values. Can be hist.Hist or tuple (vals, edges)
     - ax : the mpl Axes
     - shift : shifting x axis. if False, do nothing. If True, shift by -0.5 (for Integer axises). If float, shift by given value
     - confidenceIntervals : the uncertainties to use. If None, use sqrt(h.variances) from histogram. If 2D array then up/down uncertainties as confidence intervals
     - kwargs : passed to Axes.fill_between
    """
    if isinstance(h, hist.Hist):
        assert len(h.axes) == 1
        vals = h.values()
        edges = h.axes[0].edges
    else:
        vals = h[0]
        edges = h[1]
    if confidenceIntervals is None:
        confidenceIntervals = np.stack([vals + np.sqrt(h.variances()), vals - np.sqrt(h.variances())])
        if np.count_nonzero(confidenceIntervals<0):
            warnings.warn("Confidence interval extends below zero for some bins. You should use Poisson errors")
    if shift is not False:
        edges += + (-0.5 if shift is True else shift)

    x = []
    y1 = []
    y2 = []

    def fillUnc(bin):
        y1.append(confidenceIntervals[0][bin])
        y2.append(confidenceIntervals[1][bin])

    for i in range(len(edges)-1):
        x.append(edges[i])
        fillUnc(i)
        x.append(edges[i+1])
        fillUnc(i)

    ax.fill_between(x, y1, y2, **kwargs)
    #return x, y1, y2
