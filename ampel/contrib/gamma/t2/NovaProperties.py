#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author            : rk <ruslan.konno@desy.de>
# Date              : 10.07.2019
# Last Modified Date: 12.02.2020
# Last Modified By  : rk <ruslan.konno@desy.de>

# T2 module for filtering nova-like events based on
# 1) catalog matching
#    used white dwarf catalog derived from the GaiaDR2 catalog (Fusillo et al. 2019, MNRAS)
#    - accepts alerts within search radius (config parameter) of a cataloged white dwarf
#    - derives distance to white dwarf
# 2) lighcurve properties
#    peak magnitude should be above treshold value
#    declination time by 2 magnitudes should be below treshold value
# 3) lightcurve stats
#    chi2/N_dof should be within treshold values
#    - polyfit shouldn't be too awful
#    Test Statistic against a constant should be above treshold value
#    - want to guarantee variability and not some systematic offset

from extcats import CatalogQuery
from pymongo import MongoClient
from extcats.catquery_utils import get_closest
from ampel.base.abstract.AbsT2Unit import AbsT2Unit

import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)


class NovaProperties(AbsT2Unit):
    version = 1.01
    resources = ('extcats.reader',)

    def __init__(self, logger = None, base_config=None):

        self.logger = logger

        # all the parameters in the catalog:
        # for more info see Fusillo et al. 2019, MNRAS
        #['WD','DR2Name', 'Source', 'RAdeg', 'e_RAdeg', 'DEdeg', 'e_DEdeg', 'Plx', 'e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'epsi', 'amax', 'FG', 'e_FG', 'Gmag', 'FBP', 'e_FBP', 'BPmag', 'FRP', 'e_FRP', 'RPmag', 'E(BR/RP)', 'GLON', 'GLAT', 'Density', 'AG', 'SDSS', 'umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag'. 'e_zmag', 'Pwd', 'f_Pwd', 'TeffH', 'e_TeffH', 'loggH', 'e_loggH', 'MassH', 'e_MassH', 'chi2H', 'TeffHe', 'e_TeffHe', 'loggHe', 'e_loggHe', 'MassHe', 'e_MassHe', 'chisqHe']
        self.wd_query = CatalogQuery.CatalogQuery(
            cat_name = 'gaia2wd',
            coll_name = 'srcs',
            ra_key = 'RAdeg',
            dec_key = 'DEdeg',
            dbclient = MongoClient(base_config['extcats.reader']))

    def check_if_no_coinciding_WD(self, lc):
        '''
        Description: searches catalog for coinciding white dwarfs
        ------------
        Params:
        # lc - ampel.base.LightCurve
        ------------
        Return:
        # DR2_candidates_names [list[string]] - labels of coinciding GaiaDR2 sources
        # WD_candidates_names [listt[string]] - labels of associated white dwarves
        # matches_RA [listt[float]] - RA coordinates of coinciding white dwarves
        # matches_Dec [list[float]] - Dec coordinates of coinciding white dwarves
        # DR2_candidates_dist [list[float]] - distances [kpc] of coinciding white dwarves
        # DR2_candidates_e_dist [list[float]] - distance uncertainties [kpc] of coinciding white dwarves
        if no matches, return is empty lists
        '''

        cnd_ra, cnd_dec = lc.get_pos(ret = "latest")

        matches = self.wd_query.findwithin(cnd_ra, cnd_dec, self.search_radius, method = 'healpix', circular = False)

        if matches == None:
            return [], [], [], [], [], []

        num_matches = len(matches['WD'])

        WD_candidates_names = list(matches['WD'])
        DR2_candidates_names = list(matches['DR2Name'])

        matches_RA, matches_Dec = matches['RAdeg'], matches['DEdeg']
        matches_Plx, matches_e_Plx = matches['Plx'], matches['e_Plx']

        # get distance[kpc] from parallax[mas] in catalog
        DR2_candidates_dist = list([])
        DR2_candidates_e_dist = list([])

        for i in range(0, num_matches):
            Plx_i, e_Plx_i = matches_Plx[i], matches_e_Plx[i]

            dist_away = 1/Plx_i
            e_dist_away = e_Plx_i/(Plx_i**2)

            DR2_candidates_dist.append(dist_away)
            DR2_candidates_e_dist.append(e_dist_away)

        return WD_candidates_names, DR2_candidates_names, matches_RA, matches_Dec, DR2_candidates_dist, DR2_candidates_e_dist

    def check_stats(self, date, mag, e_mag, polfit, controlfit):

        '''
        Description: calculates Chi2/N_dof and TS
        ------------
        Params:
        # date - observation dates
        # mag - corresponding magnitudes
        # e_mag - uncertainties of corresponding magnitudes
        # polfit - polynomial fit to test
        # controlfit - constant fit null hypothesis
        ------------
        Return:
        # chi2/N_dof [float] - chi2 per degree of freedom of polynomial fit
        # TS [float] - test statistic of polynomial fit vs. constant
        '''

        # polyfit test hyphothesis
        fit = np.poly1d(polfit)

        # constant null hypothesis
        null_fit = np.poly1d(controlfit)

        # ---- Chi2 ----
        chi2_dof = sum(np.array(mag-fit(date))**2/np.array(e_mag)**2)/(len(mag)-self.poly_deg)

        # ---- TS -----
        log_like_poly5D = -sum(np.array(mag-fit(date))**2/(2*np.array(e_mag)**2))
        log_like_null = -sum(np.array(mag-null_fit(date))**2/(2*np.array(e_mag)**2))
        TS = 2*(log_like_poly5D-log_like_null)

        return chi2_dof, TS

    def check_LC(self, date, mag, polfit):

        '''
        Description: get LC features by curve discussion
        -----------
        Params:
        # date - observation dates
        # mag - corresponding magnitudes
        # polfit - polynomial lightcurve fit
        -----------
        Return:
        # global_peak_mag [float] - predicted magnitude of global peak, if prediction fails then return global peak from data
        # global_peak_day [float] - predicted day [Julian Date] of global peak, if prediction fails then return day from data
        # decline_time [float] - predicted time of declination [days] by 2 mag, if prediction fails then return time from data (time between peak and mag closet to peak-2)
        # peak_predicted [bool] - True if global_peak_mag taken from fit, False if from data
        # decline_time_predicted [bool] - True if decline_time taken from fit, False if from data
        if no peak can be found, return is None for each value
        '''

        # curve discussion necessary derivatives and related extrema
        fit = np.poly1d(polfit)
        fit_deriv = fit.deriv()
        fit_deriv_deriv = fit_deriv.deriv()
        fit_deriv_roots = fit_deriv.roots
        fit_extrema = fit(fit_deriv_roots)
        fit_deriv_deriv_check = fit_deriv_deriv(fit_deriv_roots)

        #select all extrema w. imaginary part == 0
        fit_selection = [(x,y,z) for (x,y,z) in zip(fit_deriv_roots, fit_extrema, fit_deriv_deriv_check) if y.imag == 0]

        peak_mag = list([])
        peak_day = list([])

        # find all peaks and corresponding dates
        for (x,y,z) in fit_selection:
            if z.real > 0:
                peak_day.append(x.real)
                peak_mag.append(y.real)

        if len(peak_mag) < 1:
            return None, None, None, False, False

        # get global values
        global_peak_mag_idx = np.argmin(peak_mag)
        global_peak_mag = peak_mag[global_peak_mag_idx]
        global_peak_day = peak_day[global_peak_mag_idx]

        #filter for minima with realistic magnitude (absolute value not higher than 100) [edge effects]
        # 100 is sort of arbitary, I admit
        # this is for edge effects, in which the global absolute mag will go wild
        # in case it does, use data points for global peak and date
        # TODO: see if you can find a better solution if/when you get around to this part again
        # shouldn't be that hard

        peak_predicted = True
        if abs(global_peak_mag) > 100:
            peak_predicted = False
            max_from_data_idx = np.argmin(mag)
            global_peak_mag = mag[max_from_data_idx]
            global_peak_day = date[max_from_data_idx]

        # decline time for 2 mag below peak
        decline_val = global_peak_mag + 2

        pc = polfit.copy()
        pc[-1] -= decline_val

        decline_times = np.array([x.real for x in np.roots(pc) if x.imag == 0])
        decline_times -= global_peak_day

        decline_time_predicted = True

        # get correct time of decline
        # - has to be related to a magnitude after the global peak
        # - given the above, has to be closet date to peak date
        # if there are no declination times for after the peak
        # - use data (time between peak and mag closet to peak-2)

        if len(decline_times[decline_times > 0]):
            decline_time = min(decline_times[decline_times > 0])
        else:
            decline_time_predicted = False
            decline_time_from_data_idx = np.argmin(abs(np.array(mag)-decline_val))
            decline_time = abs(date[decline_time_from_data_idx] - global_peak_day)


        return global_peak_mag, global_peak_day, decline_time, peak_predicted, decline_time_predicted

    def run(self, light_curve, run_config):

        rc_dic = run_config
        self.search_radius = rc_dic['SEARCH_RADIUS']
        self.max_peak_mag = rc_dic['MAX_PEAK_MAG']
        self.max_decline_time = rc_dic['MAX_DECLINE_TIME']
        self.poly_deg = rc_dic['POLY_DEG']
        self.min_chi2_dof = rc_dic['MIN_CHI2_DOF']
        self.max_chi2_dof = rc_dic['MAX_CHI2_DOF']
        self.min_TS = rc_dic['MIN_TS']

        date = light_curve.get_values("obs_date")
        mag = light_curve.get_values("mag")
        e_mag = light_curve.get_values("sigmapsf")

        # fits for LC, polfit = Polynomial & control = constant
        # control used for Test Statistic (TS)
        polfit = np.polyfit(date, mag, self.poly_deg)
        controlfit = np.polyfit(date, mag, 0)

        # check white dwarf coincidence
        WD_name, DR2_name, WD_ra, WD_dec, WD_dist, WD_e_dist = self.check_if_no_coinciding_WD(light_curve)

        # check stats of LC
        if len(mag) > self.poly_deg:
            chi2_dof, TS = self.check_stats(date, mag, e_mag, polfit, controlfit)
        else:
            chi2_dof = 0.
            TS = 0.

        # check features of LC
        peak_mag, time_of_peak, decline_time, peak_predicted, decline_time_predicted = self.check_LC(date, mag, polfit)

        # accepted bool for quick check during T3 if alert should be triggered on
        accepted = True

        # catalogue match check
        if not WD_name:
            accepted = False

        # statistics and LC checks

        if self.min_chi2_dof > chi2_dof or chi2_dof > self.max_chi2_dof:
            accepted = False

        if TS < self.min_TS:
            accepted = False

        if peak_mag == None or self.max_peak_mag < peak_mag:
            accepted = False

        if decline_time == None or self.max_decline_time < decline_time:
            accepted = False

        return {
            "accepted": accepted,
            "polyfit": list(polfit),
            "WD_name": list(WD_name),
            "WD_ra": list(WD_ra),
            "WD_dec": list(WD_dec),
            "DR2_name": list(DR2_name),
            "WD_dist": list(WD_dist),
            "WD_e_dist": list(WD_e_dist),
            "peak_mag": peak_mag,
            "time_of_peak": time_of_peak,
            "decline_time": decline_time,
            "peak_predicted": peak_predicted,
            "decline_time_predicted": decline_time_predicted,
            "chi2_dof": chi2_dof,
            "TS": TS,
        }
