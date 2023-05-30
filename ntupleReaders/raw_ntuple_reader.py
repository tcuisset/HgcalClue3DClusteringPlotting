import os

import uproot
import numpy as np
import awkward as ak


class RawNtupleReader:
    def __init__(self, datatype, ntupleNumber, beamEnergy=None) -> None:
        #os.environ["X509_USER_PROXY"] = "~/.t3/proxy.cert"
        self.path = os.path.join("root://eoscms.cern.ch//", "eos/cms/store/group/dpg_hgcal/tb_hgcal/2018/cern_h2_october/offline_analysis")
        if datatype == "data":
            self.path = os.path.join(self.path, "ntuples/v16", f"ntuple_{ntupleNumber}.root")
        else:
            self.path = os.path.join(self.path, "sim_ntuples/CMSSW11_0_withAHCAL_newBeamline/FTFP_BERT_EMN", datatype,
                f"electrons/ntuple_sim_config22_pdgID11_beamMomentum{beamEnergy}_listFTFP_BERT_EMN_0000_{ntupleNumber}.root")
    
    def getFile(self):
        return uproot.open(self.path)

    def getHitsTree(self):
        return self.getFile()["rechitntupler/hits"]

def getAllRawNtupleReaders(datatype, beamEnergy):
    if datatype == "data":
        # all_ntuples = [435, 436, 437, 439, 441, 442, 443, 444, 447, 450, 451, 452, 453, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477, 479, 480, 481, 482, 483, 484, 486, 487, 489, 490, 491, 493, 494, 495, 496, 501, 502, 503, 504, 505, 506, 507, 508, 509, 594, 595, 596, 597, 599, 601, 603, 604, 606, 607, 608, 609, 610, 611, 613, 614, 616, 617, 618, 619, 620, 621, 622, 635, 636, 637, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 652, 653, 654, 655, 656, 657, 659, 661, 663, 664, 665, 666, 667, 671, 672, 673, 674, 675, 676]
        beamEnergyToNtuples = {
            20.0: [436, 437, 439, 441, 442, 443, 444, 447, 450, 451, 452, 453, 455],
            30.0: [594, 595, 596, 597, 599, 601, 603, 604, 606, 607],
            50.0: [456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 608, 609, 610, 611, 613, 614, 616, 617, 618, 619],
            80.0: [466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 655, 656, 657, 659, 661, 663],
            100.0: [477, 479, 480, 481, 482, 483, 484, 486, 487, 489, 490, 491],
            120.0: [620, 621, 622, 635, 636, 637, 639, 640, 641, 642, 643, 644],
            150.0: [493, 494, 495, 496, 501, 502, 503, 504, 505, 506, 507, 508, 509],
            200.0: [664, 665, 666, 667, 671, 672, 673, 674, 675, 676],
            250.0: [645, 646, 647, 648, 649, 650, 652, 653, 654],
            300.0: [435]
        }
        ntuples = beamEnergyToNtuples[beamEnergy]
    else:
        ntuples = [0, 1, 2, 3]
    
    return [RawNtupleReader(datatype, ntupleNumber, beamEnergy) for ntupleNumber in ntuples]


def apply_dEdx(df):
    """ Apply dEdX to compute energy in MeV from dataframe of raw ntuples 
    Parameters : 
     - df : dataframe holding at least rechit_layer and rechit_energy (in MIP)
    """
    # Map of layer-1 to dEdX (first value is for layer 1)
    # ie map_dedx[0] is for layer 1
    map_dedx = [11.289,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,9.851,11.360,11.360,11.360,11.360,10.995,10.995,11.153,6.17]
    
    # Trick taken from https://stackoverflow.com/a/16993364
    layers, inv_map = np.unique(df.rechit_layer, return_inverse=True)
    dedx_array = np.array([map_dedx[layer-1] for layer in layers])[inv_map]

    return df.assign(rechit_energy_MeV=df.rechit_energy*dedx_array)



# zip into records
#record = event_data[0][["rechit_energy", "rechit_layer"]]
#energy_layer = ak.zip({key : record[key] for key in record.fields})