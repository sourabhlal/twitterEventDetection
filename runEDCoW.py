#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, math
import math
import pywt # pip install PyWavelets
import igraph # pip install python-igraph
import numpy as np
from collections import Counter
from tqdm import tqdm

from EDCoW import helperFunctions as edcow

from helper import preprocessing
from helper import readTweets

# Get all tweets
if len(sys.argv) == 2:
	print ("Running EDCoW on "+sys.argv[1])
	dataset = sys.argv[1]
else:
	print ("Running EDCoW on data/manchester_attack.csv")
	dataset = 'data/manchester_attack.csv'


tweetList, t1_time, t2_time = readTweets.getTweets(dataset)

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 60

# Group the tweets
tweetBuckets,_ = readTweets.tweetsToBuckets(tweetList,bucketSize,t1_time,t2_time)

cur_time = len(tweetBuckets) 

# Build the bag of words (after preprocessing)
words = edcow.get_words(tweetBuckets)

# Signal for each word
signal_dict = dict()

# Sliding window size for stage 2 signal:  1 hour
delta = 4
_lambda = 11.0

print("Calculating stage 1 and stage 2 signal and filtering them, no of words:", len(words))
num_words = len(words)

auto_corr = np.zeros(num_words)
for i, word in enumerate(words):
	sys.stdout.write('\r Calculating auto corr, words:%d out of %d' % (i, num_words))
	stage1_sig = [edcow.s_w(tweetBuckets, word, t , cur_time) for t in range(cur_time)]
	stage2_sig = edcow.second_stage(stage1_sig, delta)

	# Zero time lag auto_corr: basically a dot product	
	auto_corr[i] = np.dot(stage2_sig, stage2_sig)
	signal_dict[word] = stage2_sig

median_acc = np.median(auto_corr)
mid_abs_dev = np.median(np.abs([(auto_corr[i] - median_acc) for i in range(num_words)]))

theta_1 = median_acc + _lambda * mid_abs_dev

# Filter away signals
for i, word in enumerate(words):
	if auto_corr[i] <= theta_1:
		del signal_dict[word]

# Filtered
words = list(signal_dict.keys())
print("Words left after filtering 1:", len(words))

print("Computing the cross correlation")
# Zero time lag cross correlation = dot product
cross_corr = np.zeros(shape=(len(words), len(words)))
for i in range(len(words)):
	for j in range(i+1, len(words)):
		cross_corr[i,j] = np.dot(signal_dict[words[i]], signal_dict[words[j]])
		cross_corr[j,i] = cross_corr[i,j]

print("Filtering with theta2")
# Set the elements less than theta2 to zero
for i in range(len(words)):
	median_cross_corr = np.median(cross_corr[i,:]) 
	theta_2 = median_cross_corr + _lambda * edcow.median_abs_dev(cross_corr[i,:], median_cross_corr)
	for j in range(len(words)):
		if cross_corr[i,j] <= theta_2:
			cross_corr[i,j] = 0

print("Computing the clusters")	
# Clustering
g = edcow.get_graph(cross_corr, range(len(words)))

print ("Making communities")
_, comm = edcow.get_communities(g)  #THIS NEEDS FIXING

print("Numbers of clusters:", len(comm))	
good_clusters  = []
for cluster in comm:		
	n = float(len(cluster))
	print("Cluster length:", n)	
	deg_c = 0.0
	for i in cluster:
		for j in cluster:
			deg_c += cross_corr[i,j]
	if n < 50:
		eps = deg_c * math.exp(1.5*n) / math.factorial(2.0*n)  #THIS NEEDS FIXING
		if eps >= 0.1:
			good_clusters.append(cluster)
	print("Clustered Words:", [words[i] for i in cluster])
