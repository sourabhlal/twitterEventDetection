#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import defaultdict
from sklearn.mixture import GaussianMixture

from featureTrajectories import featureIdentification as fi

def trainGMM(feat_traj):

    params = defaultdict(dict)

    # iterate over each word feature
    for feature, featureTrajectory in feat_traj.items():

        # derive the number of classes
        T = len(featureTrajectory)
        dominantPeriod,dominantPowerSpectrum = fi.spectral_analysis_for_dominant_period(featureTrajectory)

        n_classes = int(math.floor(float(T) / dominantPeriod))

        # create the matrix
        matrix = np.array(featureTrajectory)[np.newaxis].T

        # initialize the mixture of Gaussians
        clf = GaussianMixture(n_components=n_classes, covariance_type='full', max_iter=20, random_state=0)

        # train
        clf.fit(matrix)

        # store
        params[feature]['weights'] = clf.weights_[0]
        params[feature]['means'] = clf.means_[0][0]
        params[feature]['covariances'] = clf.covariances_[0][0][0]
    
    return params