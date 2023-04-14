import uproot
import functools

# Nominal beam energies
beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]

# Map nominal beam energy -> final beam energy, ie energy of particles in front of detector after synchrotron losses in test beam
synchrotronBeamEnergiesMap:dict[float, float] = {20 : 20, 30 : 30, 50 : 49.99, 80 : 79.93, 100 : 99.83, 120 : 119.65, 150 : 149.14, 200 : 197.32, 250 : 243.61, 300 : 287.18}

layerToZMapping = {1: 13.8774995803833,
 2: 14.767499923706055,
 3: 16.782499313354492,
 4: 17.672500610351562,
 5: 19.6875,
 6: 20.577499389648438,
 7: 22.6924991607666,
 8: 23.582500457763672,
 9: 25.697500228881836,
 10: 26.587499618530273,
 11: 28.702499389648438,
 12: 29.592500686645508,
 13: 31.50749969482422,
 14: 32.397499084472656,
 15: 34.3125,
 16: 35.20249938964844,
 17: 37.11750030517578,
 18: 38.00749969482422,
 19: 39.92250061035156,
 20: 40.8125,
 21: 42.907501220703125,
 22: 44.037498474121094,
 23: 46.412498474121094,
 24: 47.54249954223633,
 25: 49.68199920654297,
 26: 50.6879997253418,
 27: 52.881500244140625,
 28: 53.903499603271484}

# Value of threshold to compute log-weights of hits energies (weight = max(0; thresholdW0 + ln(E/(totalE)))) (in GeV)
thresholdW0 = 2.9 

@functools.lru_cache()
def loadClueParameters(clueClustersFile:str):
    return (
        uproot.open(clueClustersFile)
        ["clueParams"]
        .members
    )

@functools.lru_cache()
def loadClue3DParameters(clueClustersFile:str):
    return (
        uproot.open(clueClustersFile)
        ["clue3DParams"]
        .members
    )
