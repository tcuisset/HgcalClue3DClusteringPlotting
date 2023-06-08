import hist


def convert2DHistogramToDictOfHistograms(h:hist.Hist) -> dict[int, hist.Hist]:
    return {beamEnergy : h[{"beamEnergy" : hist.loc(beamEnergy)}] for beamEnergy in h.axes["beamEnergy"]}
