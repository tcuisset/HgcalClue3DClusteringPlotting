import pandas as pd


def makeDashLink(beamEnergy, ntuple, event):
    return f"https://hgcal-tb18-clue3d-visualization.web.cern.ch/?beamEnergy={int(beamEnergy)}&ntuple={int(ntuple)}&event={int(event)}"

def makeCsvRow(beamEnergy, ntuple, event, *, source:str, layer:int=None):
    if layer is None:
        layerStr = ""
    else:
        layerStr = str(layer)
    return f"{int(beamEnergy)};{int(ntuple)};{int(event)};{layerStr};{source}"

def printCsvRowsFromDf(df:pd.DataFrame, source:str, layerColumn=None):
    colList = ["event", "ntupleNumber", "beamEnergy"]
    if layerColumn is not None:
        colList += [layerColumn]
    for row in df[colList].itertuples():
        if layerColumn is None:
            layer=None
        else:
            layer = getattr(row, layerColumn)
        print(makeCsvRow(row.beamEnergy, row.ntupleNumber, row.event, source=source, layer=layer))