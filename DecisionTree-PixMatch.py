#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:07:37 2020

@author: yuanhanglin
"""

#%%
from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import scipy
from itertools import combinations
import sklearn.discriminant_analysis, sklearn.metrics
from sklearn.linear_model import LinearRegression
import sklearn.discriminant_analysis, sklearn.metrics, sklearn.ensemble
from sklearn.tree import DecisionTreeRegressor

#%%
def voting(predict, target, polygon_name):
    # find out all unique polygons
    unique_polygon = np.unique(polygon_name)
    
    # initialize two output arrays
    polygon_predict = np.zeros(unique_polygon.shape).astype(np.int)
    polygon_target = np.zeros(unique_polygon.shape).astype(np.int)

    for i, p in enumerate(unique_polygon):
        # find out all pixels of polygon p
        indices = np.where(polygon_name == p)[0]
        
        # find out prediction by voting, return the smallest if there's a tie
        polygon_predict[i] = scipy.stats.mode(predict[indices])[0]

        # set the target
        polygon_target[i] = target[indices[0]]

    accuracy = np.round(sklearn.metrics.accuracy_score(polygon_predict, 
                                                       polygon_target), 2)
    
    return accuracy
    
def get_spectra_polygonname_coordinate(date, path = "../data/"):
    data = loadmat(path + "auxiliary_info.mat", squeeze_me = True)
    bbl = data["bbl"]
    
    data = pd.read_csv(path + date + "_AVIRIS_speclib_subset_spectra.csv")
    polygon_names = data['PolygonName'].values
    spectra = data.values[:,5:]
    spectra = spectra[:, bbl == 1] # remove water bands
    spectra = spectra / 10000
    spectra = spectra[:, 2:] # first two bands are zero bands, 176 to 174
    spectra[spectra < 0] = 0
    spectra[spectra > 1] = 1 
    
    coordinates = pd.read_csv(path + date + "_AVIRIS_speclib_subset_metadata.csv")
    coordinates = coordinates[['X','Y']].values
    
    return spectra, polygon_names, coordinates


def construct_pair(Ta, Tb, path = "../data/"):
    Ta_spectra, Ta_polygons, Ta_coordinates = get_spectra_polygonname_coordinate(Ta, 
                                                                                 path)
    
    Tb_spectra, Tb_polygons, Tb_coordinates = get_spectra_polygonname_coordinate(Tb, 
                                                                                 path)
    
    # Step 1 : find all the polygons that are included in both Ta, Tb
    common_polygons = list(set(Ta_polygons) & set(Tb_polygons))
    
    ############### stratified sampling ####################################
    common_polygons_by_class = dict()
    for common_polygon in common_polygons:
        labels = common_polygon.split('_')[0]
        temp = common_polygons_by_class.get(labels, [])
        temp.append(common_polygon)
        common_polygons_by_class[labels] = temp
    
    polygons_for_mapping = []
    
    for key, value in common_polygons_by_class.items():
        value = np.array(value)
        value = value.astype(str)
        polygons_for_mapping += np.random.choice(value, 
                                    size = math.ceil(len(value) * 0.2),
                                    replace = False).tolist()
    ############### stratified sampling ####################################

    print("select 20% of common polygons for temporal mapping per class")

    X_Ta = []
    X_Tb = []
    
    # Step 2 : iterate all common polygons and construct pairs by checking coordinates 
    for common_polygon in polygons_for_mapping:
        train_indices = np.where(Ta_polygons == common_polygon)[0]
        test_indices = np.where(Tb_polygons == common_polygon)[0]
        
        train_pos = Ta_coordinates[train_indices, :].tolist()
        test_pos = Tb_coordinates[test_indices, :].tolist()
        
        # For each pixel, if it is included in both training and testing date, use the spectra of two dates to construct pair
        # Otherwise, discard it, i.e., not all training pixels can be used to construct pairs
        for i in range(len(test_pos)):
            try:
                idx = train_pos.index(test_pos[i])
                X_Ta.append(Ta_spectra[train_indices[idx], :])
                X_Tb.append(Tb_spectra[test_indices[i], :])
            except ValueError:
                continue
            
    X_Ta = np.vstack(X_Ta).astype('float')
    X_Tb = np.vstack(X_Tb).astype('float')

    print("Number of pixel for mapping:", len(X_Ta))
    
    return X_Ta, X_Tb, polygons_for_mapping

def helper(polygons_for_mapping, date, path = "../data/"):
    x = loadmat(path + "SusanSpectraProcessed" \
                + date + "_classesremoved.mat", squeeze_me = True)

    spectra, labels = x["spectra"], x["labels"]
    polygons = x["polygon_names"]
    helper = np.vectorize(lambda x : x in polygons_for_mapping)
    mask = helper(polygons)
    
    """
    mask 
        True  : used for pixmatch and same-date validation
        False : used for same date training / cross-date testing
    """
    
    return spectra, labels, polygons, mask

def same_date_validation(num_round, date, clf, spectra, polygons, mask, labels):
    pixel_predict = clf.predict(spectra[mask, :])
    polygon_accuracy = voting(pixel_predict, labels[mask], 
                              polygons[mask])
    print(num_round, date, "test", date, polygon_accuracy, "validation")
    return
    
def predict_and_vote(train_clf, test_spectra, test_labels, test_mask, 
                     test_polygons):
    pixel_predict = train_clf.predict(test_spectra[~test_mask, :])
    polygon_accuracy = voting(pixel_predict, test_labels[~test_mask], 
                              test_polygons[~test_mask])
    return polygon_accuracy
    
def cross_date_testing(num_round, date_pair, clf_pair, spectra_pair, 
                       polygon_pair, mask_pair, label_pair, 
                       mapping_method = None, mapping_pair = None):
    Ta, Tb = date_pair
    clf_Ta, clf_Tb = clf_pair
    Ta_spectra, Tb_spectra = spectra_pair
    Ta_polygons, Tb_polygpons = polygon_pair
    Ta_mask, Tb_mask = mask_pair
    Ta_labels, Tb_labels = label_pair
    
    if mapping_method != "no-mapping":
        Ta2Tb, Tb2Ta = mapping_pair
        if isinstance(Ta2Tb, np.ndarray):
            Tb_spectra = Tb_spectra @ Tb2Ta
            Ta_spectra = Ta_spectra @ Ta2Tb
        else:
            Tb_spectra = Tb2Ta.predict(Tb_spectra)
            Ta_spectra = Ta2Tb.predict(Ta_spectra)
        
    Tb2Ta_accuracy = predict_and_vote(clf_Ta, Tb_spectra, Tb_labels, 
                                      Tb_mask, Tb_polygons)
    Ta2Tb_accuracy = predict_and_vote(clf_Tb, Ta_spectra, Ta_labels, 
                                      Ta_mask, Ta_polygons)
    
    print(num_round, Ta, "test", Tb, Tb2Ta_accuracy, mapping_method)
    print(num_round, Tb, "test", Ta, Ta2Tb_accuracy, mapping_method)
    

#%% linear mapping
dates = ["130411", "130606", "131125",
         "140416", "140600", "140829",
         "150416", "150602", "150824"]


# generate 36 cross-date pairs by permutation 
pairs = list(combinations(dates, 2))

mapping_method = ["no-mapping", "linear-transformation", 
                  "affine-transformation", "decision-tree", "random-forest"]

import warnings
warnings.simplefilter("ignore")


"""
    Test how long does it takes to get 
    36(date pairs, choose 2 out of 9 dates) x 2(bidirectional) = 72 mapping models.
"""

mappings = []

for Ta, Tb in pairs:
    #%% select 20% of common polygons
    X_Ta, X_Tb, polygons_for_mapping = construct_pair(Ta, Tb)
    
    #%% PixMatch
    # DecisionTree transformation
    Ta2Tb_dt = DecisionTreeRegressor()
    Tb2Ta_dt = DecisionTreeRegressor()
    Ta2Tb_dt.fit(X_Ta, X_Tb)
    Tb2Ta_dt.fit(X_Tb, X_Ta)