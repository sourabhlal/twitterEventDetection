import sys, csv, math
from collections import defaultdict

def getTweets(filename):
	"""
	Extracts tweets from csv file
	:param filename: name of csv file that contains tweet information (column 1 has time, columnn 2 has text, no title)
	Returns tweetList: list of dictiontaries, where each dictionary contains the time and text of 1 tweet
	Returns t1_time: timestamp of first tweet in dataset
	Returns t2_time: timestamp of last tweet in dataset
	"""
	tweetList = []
	t1_time = sys.maxsize
	t2_time = 0

	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile,quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True)
		for row in reader:
			tweetDict = defaultdict(dict)
			tweetDict["createdAtAsLong"] = int(row[0])
			tweetDict["text"] = row[1]
			tweetList.append(tweetDict)
			if tweetDict["createdAtAsLong"] < t1_time:
				t1_time = tweetDict["createdAtAsLong"]
			if tweetDict["createdAtAsLong"] > t2_time:
				t2_time = tweetDict["createdAtAsLong"]
	return tweetList, t1_time, t2_time

def tweetsToBuckets(tweets,timeStepSize,t1_time,t2_time):
	"""
	Splits list of tweets into buckets of specified size
	:param tweets: list of dictiontaries, where each dictionary contains the time and text of 1 tweet
	:param timeStepSize: bucketSize (1=seconds,60=minutes,3600=hours,86400=days)
	:param t1_time: timestamp of first tweet in tweets
	:param t1_time: timestamp of last tweet in tweets
	Returns tweetBuckets: list of lists, where each list contains all the tweets in that bucket
	Returns bucketStartTime: list that contains earliest timestamp that will be allowed into each bucket
	"""
	numberOfBuckets = int((t2_time - t1_time) / timeStepSize) + 1
	tweetBuckets = []
	bucketStartTime = []
	curr = t1_time
	for b in range(numberOfBuckets):
		tweetBuckets.append([])
		bucketStartTime.append(curr)
		curr+=timeStepSize
	for t in tweets:
		bucket = math.floor((t["createdAtAsLong"] - t1_time) / timeStepSize)
		tweetBuckets[bucket].append(t)
	return tweetBuckets, bucketStartTime