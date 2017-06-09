import csv, sys, math, string, re, time
from collections import defaultdict

from sparselsh import LSH #pip install sparselsh
import numpy as np 
from scipy.sparse import csr_matrix, vstack



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

def stem(token):
	"""
	Stems token
	:param token: token to be stemmed
	Returns token: stemmed token
	"""
	if token.endswith("ing"):
		token = token[:-3]
	elif token.endswith("ed"):
		token = token[:-2]
	elif token.endswith("es"):
		token = token[:-2]
	elif token.endswith("s") and len(token) > 3 and token[-2] in "wrtpsdfgklmnbvcz":
		token = token[:-1]
	return token

def remove_symbol_headTail(token):
	"""
	Removes symbols from the head and tail of a token (example: "***@^&!%happy!!!!!!!!!" --> "happy")
	:param token: token to be un-padded
	Returns: un-padded token
	"""
	f_index = 0
	for x in token:
		if x not in string.punctuation:
			f_index = token.index(x)
			break
	b_index = len(token)
	for i,x in enumerate(reversed(token)):
		if x not in string.punctuation:
			b_index = i
			break
	if b_index == 0:
		return token[f_index:]
	else:
		return token[f_index:-b_index]

def removeEmojis(text):
	"""
	Removes emojis from text
	:param text: string with emojis
	Returns: string without emojis
	"""
	return ''.join(c for c in text if c <= '\uFFFF')

def getTokens(tweetText,removeStopwords):
	"""
	Takes a string and tokenizes it
	:param tweetText: string to be tokenized
	:param removeStopwords: boolean flag to specify whether stopwords should be removed or not
	Returns words: list of tokens
	"""

	#remove urls, hashtags, @s from tweets
	tweetText = re.sub(r'http\S+', '', tweetText)
	tweetText = re.sub(r'@\S+', '', tweetText)
	tweetText = re.sub(r'#\S+', '', tweetText)
	tweetText = removeEmojis(tweetText)
	tweetText = tweetText.lower()

	# clean the tweets and split only the words
	words = tweetText.decode('utf-8')\
		.translate(string.punctuation).split()
	words = [str(w) for w in words]
	# stem the words & remove stopwords
	if removeStopwords:
		stopwords = ["i","a","about","an","and","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","this","to","was","what","when","where","who","will","with","the","www", "you", "me", "so", "my","they","your","but","i'm","he","his","if","do","it's","we","him","her","has"]
		words = [word for word in words if word not in stopwords]
	filtered_words = [word for word in words if len(word)>1]
	filtered_words = [remove_symbol_headTail(w) for w in filtered_words]
	processed_words = [stem(w) for w in filtered_words]

	# remove any empty strings
	words = [word for word in words if word not in [" ", ""]]

	# remove duplicates
	words = list(set(processed_words))
	return words


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

        for word in tweet['tokens']:
            if word == "":
                continue
            else:
                # if the word does not exist
                if word not in TFt:
                    TFt[word] = [0] * (T + 1)
                    TF[word] = 0

                # increase the frequency of the current word for day t
                TFt[word][t] += 1
                TF[word] += 1

    featTraj = {}

    for key in TFt:
        featTraj[key] = [0] * (T + 1)
        for idx, val in enumerate(TFt[key]):
            try:
                featTraj[key][idx] = (float(val) / Nt[idx]) * math.log(float(N) / TF[key])
            except:
                print ("NO DOCUMENTS ON DAY ", idx)
    return featTraj

def fsd(hashTab,newTweet,numTop=50,numLatestTweets=10,thresh=0.5):

	min_dis = 1
	idTweet = -1

	#returns total numTop  ranked results
	nnTweets = hashTab.query(newTweet,numTop)

	if len(nnTweets) > 1:

		#ids of the tweets
		ids = [elem[0][1] for elem in nnTweets]

		#sparse vector representing tweets
		vecs = [elem[0][0] for elem in nnTweets]

		#create sparse matrix
		vecs = vstack(vecs)

		#maximum dot product
		cosDist = np.squeeze(np.array(newTweet.dot(vecs.T).todense()))
		min_dis = np.max(cosDist)
		idTweet = ids[np.argmax(cosDist)]

	# if min_dis >= thresh:
	# 	latest = hashTab[-numLatestTweets:]
	# 	cosDist = np.squeeze(np.array(newTweet.dot(latest.T).todense()))
	# 	min_dis_ = np.max(cosDist)
	# 	if min_dis_ < min_dis:
	# 		idTweet = np.argmax(cosDist)
	# 		min_dis = min_dis_
	return min_dis,idTweet




#create the tf-idf for entire corpus

if len(sys.argv) == 2:
	print ("Running realTime on "+sys.argv[1])
	dataset = sys.argv[1]
else:
	print ("Running realTime on data/manchester_attack.csv")
	dataset = 'data/manchester_attack.csv'

tweetList, t1_time, t2_time = getTweets(dataset)

for t in tweetList:
	t["tokens"]=getTokens(t["text"],False)

featTraj = build_feature_trajectories(tweetList,t1_time,t2_time,60)

#number of bursts we want to track
numBursts = 10

#max number of tweets in an event
maxbucketSize = 50

#similarity threshold for filtering tweets
similarityThreshold = 0.5
buckets = [[] for x in range(numBursts)]

def getTime(i):
	tweet = indexedTweets[i]
	return tweet["createdAtAsLong"]

def getText(i):
	tweet = indexedTweets[i]
	return tweet["text"]

def printBucketStatus(idx):
	for bucket in buckets:
		if len(bucket) < 15:
			continue
		else:
			bucketTweets = [getText(x) for x in bucket]
			print (str(idx)+" EVENT = " + str(bucketTweets))
			import ipdb; ipdb.set_trace()

def bucketTweet(tweetId,nearestNeighbour,min_dis):
	for i,bucket in enumerate(buckets):
		#find bucket with nearest neighbour
		if nearestNeighbour in bucket:
			#if bucket is full, clear old half
			if len(bucket) == maxbucketSize:
				bucket = bucket[int(maxbucketSize/2):]
			#add tweet
			bucket.append(tweetId)
			return (str(tweetId)+" added to bucket "+str(i))
	#if nearest neighbour not in any bucket, put in empty bucket
	for i,bucket in enumerate(buckets):
		if bucket == []:
			#add tweet
			bucket.append(tweetId)
			return (str(tweetId)+" added to bucket "+str(i))
	#if no empty bucket clear bucket with oldest most recent tweet
	#find bucket with with lowest timestamp on most recent tweet
	lowesetTimestampOnMostRecentTweet = [max([getTime(i)] for i in bucket) for bucket in buckets]
	todelete = 0
	for i,t in enumerate(lowesetTimestampOnMostRecentTweet):
		if t == min(lowesetTimestampOnMostRecentTweet):
			todelete = i
	#empty it
	buckets[todelete] = []
	#add tweet to it
	(buckets[todelete]).append(tweetId)
	return (str(tweetId)+" added to bucket "+str(todelete))

# Hash Table created incrementally
hashTable = LSH(10,len(featTraj.keys()))

#keys corresponds to the tokens in entire corpus
keys = sorted(featTraj.keys())

#id of the tweet nearnest to incoming tweet
lshIds = []
mDIS = []
indexedTweets = {}

#initialisation for the first tweet
lshIds.append(0)
mDIS.append(0)

#loop through tweets one by one
for idx,tweet in enumerate(tweetList):
	indexedTweets[idx]=tweet
	
	if idx % 100 == 0:
		print ('Total Processed Tweets: {}'.format(idx))

	t = (int(tweet['createdAtAsLong']) - t1_time) // 60
	
	#vector representing document
	vec = [0]*len(featTraj.keys())
	for word in tweet['tokens']:
		if word == "":
			continue
		else:
			vec[keys.index(word)] = featTraj[word][t]
	#normalise 
	norm = np.linalg.norm(vec) + 1e-6
	vec /= norm

	#sparse vec for LSH lib
	sparseVec = csr_matrix(vec)

	#find the nearest tweet id , -1 if not found
	if t != 0:
		min_dis,idTweet = fsd(hashTable,sparseVec)
		lshIds.append(idTweet)
		mDIS.append(min_dis)
		if 1-min_dis < similarityThreshold:
			output = bucketTweet(idx,idTweet,min_dis)
			#print (output)
	else:
		bucketTweet(idx,0,0)

	#insert new tweet in the table
	hashTable.index(sparseVec,extra_data=idx)
	printBucketStatus(idx)
