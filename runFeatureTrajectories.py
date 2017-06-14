#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

from featureTrajectories import dataRepresentation as dr
from featureTrajectories import featureIdentification as fi
from featureTrajectories import identifyingBursts as ib
from featureTrajectories import eventsFromFeatures as ef

from helper import preprocessing
from helper import readTweets

if len(sys.argv) == 2:
	print ("Running featureTrajectories on "+sys.argv[1])
	dataset = sys.argv[1]
else:
	print ("Running featureTrajectories on data/manchester_attack.csv")
	dataset = 'data/manchester_attack.csv'

tweetList, t1_time, t2_time = readTweets.getTweets(dataset,1)

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 60

"""
USE THIS CODE INSTEAD OF LINES 15-25,56 if you want to test using the Random Attacks dataset
tweetList1, t1_time, t2_time = readTweets.getTweets('data/randomTweets.csv',5)
tweetList2, t3_time, t4_time = readTweets.getTweets('data/attackTweets.csv',1)

tweetList = tweetList1+tweetList2

if t3_time<t1_time:
	t1_time = t3_time
if t2_time<t4_time:
	t2_time = t4_time	

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 86400
FLAG = 800
"""

for t in tweetList:
	t["text"]=preprocessing.getTokens(t["text"],False)

#build feature trajectories
features, Mf = dr.build_feature_trajectories(tweetList, t1_time, t2_time, bucketSize)

#categorize features
"""
NOTE: 
To manually set the boundary between important and unimportant events, set FLAG to your selected value
If the FLAG is set to 0 or a negative number featureTrajectories will set it based on the heuristics of the stopwords.
"""
FLAG=25
HH,LH,HL,LL = fi.categorizing_features(features,FLAG)
print (HH)
#remove noisy features
for LLf in LL:
    features.pop(LLf[0],None)

# identify feature bursts
params = ib.trainGMM(features)

#form events
events = ef.unsupervised_greedy_event_detection(HH,features,params,Mf)
