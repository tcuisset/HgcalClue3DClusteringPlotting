import math
import numpy as np
from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA
from cleaning import cleaning
from moustache import runMustache


# Transform from (eta,phi) to (x,y,z)
def cart_from_etaphi(_eta, _phi):
    theta = 2*math.atan(math.exp(-_eta))
    x = math.sin(theta)*math.cos(_phi)
    y = math.sin(theta)*math.sin(_phi)
    z = math.cos(theta)

    return x,y,z


def checkGaps(_vec, _centre):
    _vec = np.unique(_vec)
    maxExcursion = 8
    minExcursion = 8
    if _centre - minExcursion < 1:
        minSlice = 1
        minValue = 1
    else:
        minSlice = _centre - minExcursion
        minValue = _centre - minExcursion
    maxSlice = _centre + maxExcursion
    if maxSlice > _vec[len(_vec)-1]:
        maxValue = _vec[len(_vec)-1]
    else:
        maxValue = _centre + maxExcursion
    ref = np.linspace(minValue,maxValue,maxValue-minValue+1)
    _vec = _vec[minSlice-1:maxSlice]
    if len(ref) == len(_vec) and (ref == _vec).all():
        return False
    else:
        return True



class event:
    def __init__(self, torchEvent, num, pu):

        # -------- sample information -------
        self.number        = num
        self.eventNumber   = int(torchEvent.event[0])
        self.sampleNumber  = int(torchEvent.event[1])
        self.etaSign       = int(torchEvent.event[2])

        # --------- gun information ---------
        self.gunEta     = float(torchEvent.gun[0])
        self.gunPhi     = float(torchEvent.gun[1])
        self.gunEn      = float(torchEvent.gun[2])
        self.gunPID     = int(torchEvent.gun[3])
        self.gunPt      = float(self.gunEn/math.cosh(self.gunEta))

        # --------- LCs information ---------
        self.LCx    = []
        self.LCy    = []
        self.LCl    = []
        self.LCz    = []
        self.LCen   = []
        self.LCt    = []
        self.LCdt   = []
        idxs        = torchEvent.cls.numpy()
        out         = []

        for p in range(len(idxs)):
            if p == 0:
                out.append(idxs[p])
            else:
                out.append( out[len(out)-1] + idxs[p] )
        out = np.insert(out,0,0)
        for pos in range( len(out) - 1 ):
            self.LCx.append(  np.array(torchEvent.x[:,0][out[pos]:out[pos+1]], dtype=float) )
            self.LCy.append(  np.array(torchEvent.x[:,1][out[pos]:out[pos+1]], dtype=float) )
            self.LCz.append(  np.array(torchEvent.x[:,2][out[pos]:out[pos+1]], dtype=float) )
            self.LCl.append(  np.array(torchEvent.x[:,3][out[pos]:out[pos+1]], dtype=int)   )
            self.LCen.append( np.array(torchEvent.x[:,6][out[pos]:out[pos+1]], dtype=float) )
#             self.LCt.append( np.array(torchEvent.t[:,0][out[pos]:out[pos+1]], dtype=float) )
#             self.LCdt.append( np.array(torchEvent.t[:,1][out[pos]:out[pos+1]], dtype=float) )

        self.LCx   = np.array(self.LCx,  dtype=object)
        self.LCy   = np.array(self.LCy,  dtype=object)
        self.LCl   = np.array(self.LCl,  dtype=object)
        self.LCz   = np.array(self.LCz,  dtype=object)
        # self.LCen  = np.array(self.LCen, dtype=object) #This should be left commented in order to avoid conflicts in the Analyser
        # self.LCt   = np.array(self.LCt,  dtype=object)
        # self.LCdt  = np.array(self.LCdt, dtype=object)


        # --------- nTracksters ---------
        self.nTracksters = len(torchEvent.cls)

        # --------- genMatching ---------
        if pu:
            self.genMatching = torchEvent.genMatched
            if self.gunEta<0:
                self.genMatching = torchEvent.genMatched
            else:
                self.genMatching = torchEvent.genMatched - torchEvent.transition_pos_neg
        else:
            # Using this approach, when we use noPU samples the gen-matching condition is always satisfied
            self.genMatching = [i for i in range(len(self.LCx))]
        self.genMatching = np.array(self.genMatching, dtype=int)

        # --------- multi_clus ---------
        self.multi_eta   = []
        self.multi_phi   = []
        self.multi_en    = []
        self.multi_pt    = []
        self.multi_x     = []
        self.multi_y     = []
        self.multi_z     = []
        self.multi_t     = []
        self.multi_index = []
        for trk in range(len(torchEvent.multi)):
            eta     = torchEvent.multi[trk][0]
            phi     = torchEvent.multi[trk][1]
            en      = torchEvent.multi[trk][2]
            pt      = en/math.cosh(eta)
            x       = torchEvent.multi[trk][3]
            y       = torchEvent.multi[trk][4]
            z       = torchEvent.multi[trk][5]
            # x, y, z = cart_from_etaphi(eta, phi)

            self.multi_eta.append( float(eta) )
            self.multi_phi.append( float(phi) )
            self.multi_en.append( float(en) )
            self.multi_pt.append( float(pt) )
            self.multi_x.append( float(x) )
            self.multi_y.append( float(y) )
            self.multi_z.append( float(z) )
#             self.multi_t.append( float(torchEvent.timing[trk]) )
            self.multi_index.append( trk )

        self.multi_eta   = np.array(self.multi_eta)
        self.multi_phi   = np.array(self.multi_phi)
        self.multi_en    = np.array(self.multi_en)
        self.multi_pt    = np.array(self.multi_pt)
        self.multi_x     = np.array(self.multi_x)
        self.multi_y     = np.array(self.multi_y)
        self.multi_z     = np.array(self.multi_z)

        # --------- PCA and energy ratio for quality trackster ---------
        self.wpca              = []
        self.wpca_explVar      = []
        self.wpca_explVarRatio = []

        self.EnRatio_maxE      = []
        self.EnRatio_sumE      = []
        maxElc                 = max(self.multi_en)
        sumElc                 = sum(self.multi_en)

        self.skim              = []
        self.gaps              = []

        for trk in range(self.nTracksters):
            if len(np.unique(self.LCl[trk][self.LCl[trk]<26])) >= 3: #Require at least a shower spanning 3 layers in the EM part
                pca = WPCA(n_components=3)
                pca.fit(np.stack((self.LCx[trk], self.LCy[trk], self.LCz[trk])).T, weights = np.stack((self.LCen[trk], self.LCen[trk], self.LCen[trk])).T)
                self.wpca.append(pca.components_[0])
                self.wpca_explVar.append(pca.explained_variance_[0])
                self.wpca_explVarRatio.append(pca.explained_variance_ratio_[0])
            else:
                self.wpca.append(-1.)
                self.wpca_explVar.append(-1.)
                self.wpca_explVarRatio.append(-1.)
            self.EnRatio_maxE.append(self.multi_en[trk]/maxElc)
            self.EnRatio_sumE.append(self.multi_en[trk]/sumElc)
            self.gaps.append(checkGaps(self.LCl[trk], self.LCl[trk][np.argmax(self.LCen[trk])]))

            # Skimmer
            # * explained variance less than 0.95 if en_trk > 50, explained variance less than 0.92 if en_trk < 50
            # * at least 3 layers in the EM part (conditions required above)
            # * the trk should have an E > 2 GeV (Energy of a MIP crossing the entire HGCAL)

            if self.multi_en[trk] > 50:
                explVarRatio_cut = 0.95
            else:
                explVarRatio_cut = 0.92
            self.skim.append( self.wpca_explVarRatio[trk] < explVarRatio_cut or self.multi_en[trk] <= 2 ) #TRUE: to be removed; FALSE: to be kept


        self.wpca              = np.array(self.wpca,              dtype=object)
        self.wpca_explVar      = np.array(self.wpca_explVar,      dtype=object)
        self.wpca_explVarRatio = np.array(self.wpca_explVarRatio, dtype=object)
        self.EnRatio_maxE      = np.array(self.EnRatio_maxE,      dtype=object)
        self.EnRatio_sumE      = np.array(self.EnRatio_sumE,      dtype=object)
        self.skim              = np.array(self.skim,              dtype=object)


        # --------- simHits ---------
        self.simHit_x  = torchEvent.simrechits[0]
        self.simHit_y  = torchEvent.simrechits[1]
        self.simHit_z  = torchEvent.simrechits[2]
        self.simHit_l  = torchEvent.simrechits[3]
        self.simHit_en = torchEvent.simrechits[4]

        # --------- recHits ---------
        self.recHit_x  = torchEvent.rechits[0]
        self.recHit_y  = torchEvent.rechits[1]
        self.recHit_z  = torchEvent.rechits[2]
        self.recHit_l  = torchEvent.rechits[3]
        self.recHit_en = torchEvent.rechits[4]

        # -------- simTracks --------
        self.simTrack_ox = torchEvent.simTracks[0]
        self.simTrack_oy = torchEvent.simTracks[1]
        self.simTrack_oz = torchEvent.simTracks[2]
        self.simTrack_fx = torchEvent.simTracks[3]
        self.simTrack_fy = torchEvent.simTracks[4]
        self.simTrack_fz = torchEvent.simTracks[5]
        self.simTrack_HGCAL = torchEvent.simTracks[6]
        self.simTrack_PID = torchEvent.simTracks[7]
        self.simTrack_eta = torchEvent.simTracks[8]
        self.simTrack_phi = torchEvent.simTracks[9]
        self.simTrack_pt = torchEvent.simTracks[10]
        self.simTrack_energy = torchEvent.simTracks[11]
        self.simTrack_mother = torchEvent.simTracks[12]
        self.simTrack_fbrem = torchEvent.simTracks[13]
        self.simTrack_trkx = torchEvent.simTracks_trk[0]
        self.simTrack_trky = torchEvent.simTracks_trk[1]
        self.simTrack_trkz = torchEvent.simTracks_trk[2]

        self.simTracks_boundary_x = torchEvent.simTracks_boundary[0]
        self.simTracks_boundary_y = torchEvent.simTracks_boundary[1]
        self.simTracks_boundary_z = torchEvent.simTracks_boundary[2]
        self.simTracks_boundary_en = torchEvent.simTracks_boundary[3]
        self.simTracks_boundary_pt = torchEvent.simTracks_boundary[4]
        self.simTracks_boundary_eta = torchEvent.simTracks_boundary[5]
        self.simTracks_boundary_phi = torchEvent.simTracks_boundary[6]
        self.simTracks_boundary_id = torchEvent.simTracks_boundary[7]
        self.simTracks_surface_x = torchEvent.simTracks_surface[0]
        self.simTracks_surface_y = torchEvent.simTracks_surface[1]
        self.simTracks_surface_z = torchEvent.simTracks_surface[2]
        self.simTracks_HGCALfromBoundary_x = torchEvent.simTracks_HGCALfromBoundary[0]
        self.simTracks_HGCALfromBoundary_y = torchEvent.simTracks_HGCALfromBoundary[1]
        self.simTracks_HGCALfromBoundary_z = torchEvent.simTracks_HGCALfromBoundary[2]

        # --------- cleaning ---------
        self.cleanedLCx = []
        self.cleanedLCy = []
        self.cleanedLCl = []
        self.cleanedLCz = []
        self.cleanedLCen = []
        # self.cleanedLCt = []
        # self.cleanedLCdt = []
        self.pca = []
        self.pca_sub = []
        self.pca_subsub = []
        self.pca_signed = []
        self.pca_signed_sub = []
        self.pca_signed_subsub = []
        self.pca_origin = []
        for i in range(len(self.LCx)):
            if self.skim[i] == False:
                cleanpcaarr, pca_axis, origin, pca_axis_sub, pca_axis_subsub = cleaning(np.array([[self.LCx[i][j], self.LCy[i][j], self.LCz[i][j], self.LCl[i][j], self.LCen[i][j]] for j in range(len(self.LCx[i]))]),
                                                                                        'eWeighted')
                self.cleanedLCx.append(cleanpcaarr[:,0])
                self.cleanedLCy.append(cleanpcaarr[:,1])
                self.cleanedLCl.append(cleanpcaarr[:,2])
                self.cleanedLCz.append(cleanpcaarr[:,4])
                self.cleanedLCen.append(cleanpcaarr[:,3])
    #             self.cleanedLCt.append(cleanpcaarr[:,5])
    #             self.cleanedLCdt.append(cleanpcaarr[:,6])
                self.pca.append(pca_axis)
                self.pca_sub.append(pca_axis_sub)
                self.pca_subsub.append(pca_axis_subsub)
                if pca_axis[2] * self.cleanedLCz[len(self.cleanedLCz)-1][0] < 0:
                    pca_axis = [(-1) * i for i in pca_axis]
                self.pca_signed.append(pca_axis)
                if pca_axis_sub[2] * self.cleanedLCz[len(self.cleanedLCz)-1][0] < 0:
                    pca_axis_sub = [(-1) * i for i in pca_axis_sub]
                self.pca_signed_sub.append(pca_axis_sub)
                if pca_axis_subsub[2] * self.cleanedLCz[len(self.cleanedLCz)-1][0] < 0:
                    pca_axis_subsub = [(-1) * i for i in pca_axis_subsub]
                self.pca_signed_subsub.append(pca_axis_subsub)
                self.pca_origin.append(origin)
            else:
                self.cleanedLCx.append([])
                self.cleanedLCy.append([])
                self.cleanedLCl.append([])
                self.cleanedLCz.append([])
                self.cleanedLCen.append([])
    #             self.cleanedLCt.append(cleanpcaarr[:,5])
    #             self.cleanedLCdt.append(cleanpcaarr[:,6])
                self.pca.append([])
                self.pca_sub.append([])
                self.pca_subsub.append([])
                self.pca_signed.append([])
                self.pca_signed_sub.append([])
                self.pca_signed_subsub.append([])
                self.pca_origin.append([])

        self.pca        = np.array(self.pca,        dtype=object)
        self.pca_sub    = np.array(self.pca,        dtype=object)
        self.pca_subsub = np.array(self.pca,        dtype=object)
        self.pca_signed        = np.array(self.pca_signed, dtype=object)
        self.pca_signed_sub    = np.array(self.pca_signed_sub, dtype=object)
        self.pca_signed_subsub = np.array(self.pca_signed_subsub, dtype=object)


        # --------- moustache ---------
        # sel = self.skim.astype(bool) == False
        # self.moustache = runMustache(np.array([self.multi_eta[sel], self.multi_phi[sel], self.multi_en[sel], self.multi_pt[sel], np.array(self.multi_index)[sel]]))
