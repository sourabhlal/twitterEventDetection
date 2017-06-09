#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm

from signiTrend import signiTrend as st

from helper import preprocessing
from helper import readTweets

if len(sys.argv) == 2:
	print ("Running SigniTrend on "+sys.argv[1])
	dataset = sys.argv[1]
else:
	print ("Running SigniTrend on data/manchester_attack.csv")
	dataset = 'data/manchester_attack.csv'

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 60

#signiTrend properties
window_size = 4
hash_table_bits = 10
hash_function_count = 4
bias = 0.1
alerting_threshold = 0.2

#getTweets and separate to buckets based on epoch
tweetList, t1_time, t2_time = readTweets.getTweets(dataset)
tweetBuckets,bucketStartTime = readTweets.tweetsToBuckets(tweetList,bucketSize,t1_time,t2_time)

#initialize signiTrend detector
detector = st.SigniTrend(window_size = window_size,
						 hash_table_bits = hash_table_bits,
				 		 hash_function_count = hash_function_count,
				 		 bias = bias,
				 		 alerting_threshold = alerting_threshold)

"""
bucketSegment can be used when testing to run signiTrend on part of the database.
For example:
	bucketSegment = tweetBuckets[5:]  #exclude first 5 buckets
	bucketSegment = tweetBuckets[-5:] #only run on last 5 buckets
"""
bucketSegment = tweetBuckets[:]

tweet_id = 0
for index,timeStep in tqdm(enumerate(bucketSegment)):
	for tweet in timeStep:
		#tokenize tweet text & pre-processing
		tweet_tokens = preprocessing.getTokens(tweet["text"],True)

		#index tweet
		detector.index_new_tweet(str(tweet_id), tweet_tokens)
		tweet_id += 1

	#output new trends
	trending_topics = detector.end_of_day_analysis()
	tt = list(reversed(sorted(trending_topics, key=lambda k: k[1])))
	for x in tt:
		print (x)
	
	#go to next epoch
	detector.next_epoch()