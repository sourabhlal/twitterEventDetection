#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

def build_feature_trajectories(tweets, firstEpochTime, lastEpochTime, bucketSize):
    """ Build a vector of tf-idf measures in every time points
    for all word features"""

    # The tweets are represented as a list of dictionaries
    # T is the defined period

    # delta
    T = (lastEpochTime - firstEpochTime) // bucketSize

    # local Term-Frequency for each word feature
    # map of word feature to list, where the list is having T elements
    TFt = {}

    # global term frequency, total number of documents containing each feature
    TF = {}

    #feature-documentlists
    Mf = {}

    # number of documents for day t
    Nt = [0] * (T + 1)

    # total number of documents
    N = len(tweets)

    # iterate over the tweets
    tweetID = 0
    for tweet in tweets:
        tweetID+=1

        # convert the timestamp
        t = (int(tweet['createdAtAsLong']) - firstEpochTime) // bucketSize

        # increase the number of documents for day t
        Nt[t] += 1

        for word in tweet['text']:
            if word == "":
                continue
            else:
                # if the word does not exist
                if word not in TFt:
                    TFt[word] = [0] * (T + 1)
                    TF[word] = 0
                    Mf[word] = []

                # increase the frequency of the current word for day t
                TFt[word][t] += 1
                TF[word] += 1
                Mf[word].append(tweetID)

    featTraj = {}

    for key in TFt:
        featTraj[key] = [0] * (T + 1)
        for idx, val in enumerate(TFt[key]):
            try:
                featTraj[key][idx] = (float(val) / Nt[idx]) * math.log(float(N) / TF[key])
            except:
                print ("NO DOCUMENTS ON DAY ", idx)
    return featTraj, Mf