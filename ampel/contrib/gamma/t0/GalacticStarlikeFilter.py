#!/usr/bin/env python
# Author            : r. konno <ruslan.konno@desy.de>
# Date              : 20.06.2019
# Last Modified Date: 13.12.2019
# Last Modified By  : rk

# T0 module for filtering based on
# 1) image quality
# 2) galactic coordinates (latitude for now)
#    alerts below latitude treshold pass
# 3) star-galaxy classification
#    alerts above star-galaxy-score treshold pass
# 4) catalog matching, at least one specific star type within search radius
# ---- so far supported [abbrv to pass]: White Dwarfs [WD],

from ampel.base.abstract.AbsAlertFilter import AbsAlertFilter
from astropy.coordinates import SkyCoord
import numpy as np
import logging

from extcats import CatalogQuery
from pymongo import MongoClient

class GalacticStarlikeFilter(AbsAlertFilter):
    version = 0.5
    resources = ('extcats.reader',)

    def __init__(self, on_match_t2_units = None, base_config=None, run_config=None, logger= None):

        if run_config is None:
            raise ValueError("No run configuration provided.")
        if on_match_t2_units is None:
            raise ValueError("No t2 unit list provided.")

        self.logger = logger if logger is not None else logging.getLogger()
        self.on_match_t2_units = on_match_t2_units

        rc_dic = run_config

        #-------data purity cut parameters-------

        self.min_rb = rc_dic['MIN_RB']
        self.max_nbad = rc_dic['MAX_NBAD']
        self.max_fwhm = rc_dic['MAX_FWHM']
        self.max_elong = rc_dic['MAX_ELONG']
        self.max_magdiff = rc_dic['MAX_MAGDIFF']
        self.min_ndet = rc_dic['MIN_NDET']

        #-------science cut parameters-----------

        self.min_sgscore = rc_dic['MIN_SGSCORE']
        self.max_gal_lat = rc_dic['MAX_GAL_LAT']
        self.star_type = rc_dic['STAR_TYPE']
        self.search_radius = rc_dic['SEARCH_RADIUS']

        self.make_cat_query = False

        if self.star_type: # check if star type specified
            supported_star_types_and_cats = {'WD' : 'gaia2wd', }
            supported_star_types = [*supported_star_types_and_cats]

            if self.star_type in supported_star_types: # check if specified star type is supported

                self.cat_query = CatalogQuery.CatalogQuery(
                    cat_name = supported_star_types_and_cats[self.star_type],
                    coll_name = 'srcs',
                    ra_key = 'RAdeg', #TODO: ra, dec names might differ in other cats
                    dec_key = 'DEdeg',
                    dbclient = MongoClient(base_config['extcats.reader']))

                self.make_cat_query = True

            else:
                raise ValueError("Specified star type not supported.")

    def check_if_outside_gal_plane(self, candidate):

        '''
        Description:
        # checks if alert is outside galactic latitude treshold
        --------------
        Params:
        # candidate - AmpelAlert.pps[0]
        --------------
        Return:
        # bool on wether alert is above galactic latitude treshold
        '''

        coord = SkyCoord(candidate['ra'],candidate['dec'], unit='deg')

        gal_lat = coord.galactic.b.deg

        if abs(gal_lat) > self.max_gal_lat:
            return True
        else:
            return False

    def check_if_galaxylike(self, candidate):

        '''
        Description:
        # checks if alert is more galaxy-like than star-like wrt. score treshold value
        --------------
        Params:
        # candidate - AmpelAlert.pps[0]
        --------------
        Return:
        # bool on wether all 3 star-galaxy-scores are below treshold value
        '''

        sg_scores = np.array([candidate['sgscore1'], candidate['sgscore2'], candidate['sgscore3']])
        all_sgscores_below_treshold = all(score < self.min_sgscore for score in sg_scores)

        if all_sgscores_below_treshold:
            return True
        else:
            return False

    def check_if_no_match_in_cat(self, candidate):
        '''
        # Does a quick check if there are no specific stars within search radius self.search_radius
        # matches against a catalog
        -----
        Params:
        # candidate - AmpelAlert.pps[0]
        -------
        Return:
        # Return is a bool on if there are no specific stars in search radius
        # True - there are none
        # False - there is at least one
        '''

        cnd_ra, cnd_dec = candidate['ra'], candidate['dec']
        matches = self.cat_query.findwithin(cnd_ra, cnd_dec, self.search_radius, method = 'healpix', circular = False)

        if matches == None:
            return True
        else:
            return False

    def apply(self, ampel_alert):

        candidate = ampel_alert.pps[0]

        #-------data purity cut-----------------

        if candidate['rb'] < self.min_rb:
            return None

        if candidate['nbad'] > self.max_nbad:
            return None

        if candidate['fwhm'] > self.max_fwhm:
            return None

        if candidate['elong'] > self.max_elong:
            return None

        if candidate['magdiff'] > self.max_magdiff:
            return None

        if candidate['ndethist'] < self.min_ndet:
            return None

        #-------science cut---------------------
        # checks are counterproofs

        if self.check_if_outside_gal_plane(candidate):
            return None

        if self.check_if_galaxylike(candidate):
            return None

        if self.make_cat_query and self.check_if_no_match_in_cat(candidate):
            return None

        return self.on_match_t2_units
