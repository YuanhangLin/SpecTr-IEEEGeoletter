#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utilities import *

# date pairs
dates = ["130411", "130606", "131125",
         "140416", "140600", "140829",
         "150416", "150602", "150824"]

path = "../data/"

# generate 36 cross-date pairs by permutation 
pairs = list(combinations(dates, 2))

mapping_method = ["no-mapping", "linear-transformation", 
                  "affine-transformation", "decision-tree", "random-forest"]

import warnings
warnings.simplefilter("ignore")

for num_round in range(10):
    for Ta, Tb in pairs:
        # select 20% of common polygons for mapping construction
        X_Ta, X_Tb, polygons_for_mapping = construct_pair(Ta, Tb, path)
        
        # linear transformation
        Ta2Tb_linear = np.linalg.lstsq(X_Ta, X_Tb, rcond = None)[0] 
        Tb2Ta_linear= np.linalg.lstsq(X_Tb, X_Ta, rcond = None)[0] 
        
        # affine transformation
        Ta2Tb_affine = LinearRegression()
        Tb2Ta_affine = LinearRegression()
        Ta2Tb_affine.fit(X_Ta, X_Tb)
        Tb2Ta_affine.fit(X_Tb, X_Ta)
        
        # decision tree
        Ta2Tb_dt = DecisionTreeRegressor()
        Tb2Ta_dt = DecisionTreeRegressor()
        Ta2Tb_dt.fit(X_Ta, X_Tb)
        Tb2Ta_dt.fit(X_Tb, X_Ta)
        
        # random forest
        Ta2Tb_rf = sklearn.ensemble.RandomForestRegressor(n_estimators = 10)
        Tb2Ta_rf = sklearn.ensemble.RandomForestRegressor(n_estimators = 10)
        Ta2Tb_rf.fit(X_Ta, X_Tb)
        Tb2Ta_rf.fit(X_Tb, X_Ta)

        #%% preprocessing                               
        polygons_for_mapping = set(polygons_for_mapping)
                    
        """
        mask 
            True  : used for pixmatch and same-date validation
            False : used for same-date training / cross-date testing
        """
        
        Ta_spectra, Ta_labels, Ta_polygons, Ta_mask = helper(polygons_for_mapping, 
                                                             Ta, path)
        Tb_spectra, Tb_labels, Tb_polygons, Tb_mask = helper(polygons_for_mapping, 
                                                             Tb, path)
        
        #%% LDA same-date training
        clf_Ta = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        clf_Tb = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        clf_Ta.fit(Ta_spectra[~Ta_mask, :], Ta_labels[~Ta_mask])
        clf_Tb.fit(Tb_spectra[~Tb_mask, :], Tb_labels[~Tb_mask])
        
        #%% LDA same-date validation
        same_date_validation(num_round, Ta, clf_Ta, Ta_spectra, 
                             Ta_polygons, Ta_mask, Ta_labels)
        
        same_date_validation(num_round, Tb, clf_Tb, Tb_spectra, 
                             Tb_polygons, Tb_mask, Tb_labels)

        #%% LDA cross-date testing w/w.o PixMatch
        date_pair = (Ta, Tb)
        clf_pair = (clf_Ta, clf_Tb)
        spectra_pair = (Ta_spectra, Tb_spectra)
        polygon_pair = (Ta_polygons, Tb_polygons)
        mask_pair = (Ta_mask, Tb_mask)
        label_pair = (Ta_labels, Tb_labels)
        
        mapping_pairs = ((None, None), 
                        (Ta2Tb_linear, Tb2Ta_linear), 
                        (Ta2Tb_affine, Tb2Ta_affine),
                        (Ta2Tb_dt, Tb2Ta_dt), 
                        (Ta2Tb_rf, Tb2Ta_rf))
        
        for method, models in zip(mapping_method, mapping_pairs):
            cross_date_testing(num_round, date_pair, clf_pair, 
                               spectra_pair, polygon_pair, mask_pair, 
                               label_pair, method, models)