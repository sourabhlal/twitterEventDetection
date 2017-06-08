import csv
from collections import defaultdict

from featureTrajectories import dataRepresentation as dr
from featureTrajectories import featureIdentification as fi
from featureTrajectories import identifyingBursts as ib
from featureTrajectories import eventsFromFeatures as ef

import preprocessing
import readTweets

def tokenize(tweet):
    tweet["text"]=preprocessing.getTokens(tweet["text"],False)

tweetList, t1_time, t2_time = readTweets.getTweets('../eventDetectionAlgorithms/data/MIB_Dataset/timed_tweets.csv')

for t in tweetList:
	tokenize(t)

#build feature trajectories
features, Mf = dr.build_feature_trajectories(tweetList, t1_time, t2_time, 86400)

#categorize features
HH,LH,HL,LL = fi.categorizing_features(features)

#remove noisy features
for LLf in LL:
    features.pop(LLf[0],None)

# identify feature bursts
params = ib.trainGMM(features)

#form events
events = ef.unsupervised_greedy_event_detection(HH,features,params,Mf)
