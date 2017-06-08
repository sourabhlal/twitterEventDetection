#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

from featureTrajectories import dataRepresentation as dr
from featureTrajectories import featureIdentification as fi
from featureTrajectories import identifyingBursts as ib
from featureTrajectories import eventsFromFeatures as ef

import preprocessing
import readTweets

if len(sys.argv) == 2:
	print ("Running featureTrajectories on "+sys.argv[1])
	tweetList, t1_time, t2_time = readTweets.getTweets(sys.argv[1])
else:
	print ("Running featureTrajectories on data/MIB_datasample.csv")
	tweetList, t1_time, t2_time = readTweets.getTweets('data/MIB_datasample.csv')

for t in tweetList:
	t["text"]=preprocessing.getTokens(t["text"],False)

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 86400

#build feature trajectories
features, Mf = dr.build_feature_trajectories(tweetList, t1_time, t2_time, bucketSize)

#categorize features
"""
NOTE: 
To manually set the boundary between important and unimportant events, set FLAG to your selected value
If the FLAG is set to 0 or a negative number featureTrajectories will set it based on the heuristics of the stopwords.
"""
FLAG=0
HH,LH,HL,LL = fi.categorizing_features(features,FLAG)

#remove noisy features
for LLf in LL:
    features.pop(LLf[0],None)

# identify feature bursts
params = ib.trainGMM(features)

#form events
events = ef.unsupervised_greedy_event_detection(HH,features,params,Mf)
