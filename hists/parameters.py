import uproot
import functools

# Nominal beam energies
beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]

# Map nominal beam energy -> final beam energy, ie energy of particles in front of detector after synchrotron losses in test beam
synchrotronBeamEnergiesMap:dict[float, float] = {20 : 20, 30 : 30, 50 : 49.99, 80 : 79.93, 100 : 99.83, 120 : 119.65, 150 : 149.14, 200 : 197.32, 250 : 243.61, 300 : 287.18}

ntupleNumbersPerBeamEnergy = {
 20: [436, 437, 439, 441, 442, 443, 444, 447, 450, 451, 452, 453, 455],
 30: [594, 595, 596, 597, 599, 601, 603, 604, 606, 607],
 50: [456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 608, 609, 610, 611, 613, 614, 616, 617, 618, 619],
 80: [466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 655, 656, 657, 659, 661, 663],
 100: [477, 479, 480, 481, 482, 483, 484, 486, 487, 489, 490, 491],
 120: [620, 621, 622, 635, 636, 637, 639, 640, 641, 642, 643, 644],
 150: [493, 494, 495, 496, 501, 502, 503, 504, 505, 506, 507, 508, 509],
 200: [664, 665, 666, 667, 671, 672, 673, 674, 675, 676],
 250: [645, 646, 647, 648, 649, 650, 652, 653, 654],
 300: [435]
}

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

class DetectorExtentData:
    """  Very approximate detector extent for data """
    radius = 7 # 7 cm radius
    depth = layerToZMapping[28] - layerToZMapping[1]  # layer 1 to 28  = 53.9-13.8
    firstLayerZ = layerToZMapping[1]
    centerX = 3.85 # x position of center of detector
    centerY = -2.53 # y position of center of detector

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
