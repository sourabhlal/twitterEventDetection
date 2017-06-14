#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as dates

from signiTrend import signiTrend as st

from helper import preprocessing
from helper import readTweets

if len(sys.argv) == 2:
	print ("Running SigniTrend on "+sys.argv[1])
	dataset = sys.argv[1]
else:
	print ("Running SigniTrend on data/manchester_attack.csv")
	dataset = 'data/manchester_attack.csv'

tweetList, t1_time, t2_time = readTweets.getTweets(dataset,1)

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 60

#signiTrend properties
window_size = 4
hash_table_bits = 10
hash_function_count = 4
bias = 0.1
alerting_threshold = 0.2

"""
#USE THIS CODE INSTEAD OF LINES 14-31 if you want to test using the Random Attacks dataset
tweetList1, t1_time, t2_time = readTweets.getTweets('data/randomTweets.csv',5)
tweetList2, t3_time, t4_time = readTweets.getTweets('data/attackTweets.csv',1)

tweetList = tweetList1+tweetList2

if t3_time<t1_time:
	t1_time = t3_time
if t2_time<t4_time:
	t2_time = t4_time	

# bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
bucketSize = 86400

#signiTrend properties
window_size = 4
hash_table_bits = 10
hash_function_count = 4
bias = 0.1
alerting_threshold = -0.5
"""

#getTweets and separate to buckets based on epoch
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
#used for plotting graphs, make sure bucketSampleTime and bucketSegment are of the same size
bucketSampleTime = bucketStartTime[:]

"""
#THIS CODE IS USED FOR PLOTTING KEY WORD SCORES
results = {}
featuresToPlot = ['france', 'iraq', 'denmark', 'tunisia', 'yemen', 'pakistan', 'nigeria', 'ukraine', 'lebanon', 'australia', 'turkey']
featuresToPlot.sort()
for c in featuresToPlot:
	results[c] = []
"""

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
	print (time.strftime('%Y-%m-%d', time.localtime(t1_time+(index*bucketSize))),[t[0] for t in tt])

	"""
	#THIS CODE IS USED FOR PLOTTING KEY WORD SCORES
	for c in featuresToPlot:
		x = [item for item in trending_topics if item[0] == c]
		if len (x) == 1:
			results[c].append(x[0][1])
			print (x)
		else:
			results[c].append(-1.0)
	"""

	#go to next epoch
	detector.next_epoch()

"""
#THIS CODE IS USED FOR PLOTTING KEY WORD SCORES
fig, ax = plt.subplots()
days = dates.epoch2num(bucketSampleTime)
lines = []
for k in sorted(results):
	x, = ax.plot_date(days,results[k], '-')
	lines.append(x)
date_fmt = '%d-%m-%y'
date_formatter = dates.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)
fig.autofmt_xdate()
ax.legend(lines,featuresToPlot)
plt.show()
"""

"""
#THIS CODE IS USED TO PLOT TERM FREQUENCIES
tf = {}
for c in featuresToPlot:
	tf[c] = []

tweet_id = 0
for index,timeStep in tqdm(enumerate(bucketSegment)):
	temp = {}
	for c in featuresToPlot:
		temp[c] = 0
	for tweet in timeStep:
		for c in featuresToPlot:
			if c in tweet["text"]:
				temp[c]+=1
	for k,v in temp.items():
		tf[k].append(v)

fig, ax = plt.subplots()
days = dates.epoch2num(bucketSampleTime)
lines = []
for k in sorted(tf):
	x, = ax.plot_date(days,tf[k], '-')
	lines.append(x)
date_fmt = '%d-%m-%y'
date_formatter = dates.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)
fig.autofmt_xdate()
ax.legend(lines,featuresToPlot)
plt.show()
"""