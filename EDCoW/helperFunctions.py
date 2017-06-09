#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, math
import math
import pywt # pip install PyWavelets
import igraph # pip install python-igraph
import numpy as np
from collections import Counter
from tqdm import tqdm

from helper import preprocessing
from helper import readTweets

def shannon_entropy(distribution):
	sum = 0.0
	for p in distribution:
		try:
			sum -= p * (math.log(p) / math.log(2))
		except ValueError:
			print("dist:", distribution)
	return sum

def wavelet_energy(coeff):
	# Squared norm
    return np.sum(np.array(coeff) ** 2)


def h_measure(signal):
 	# Discrete wavelet transformation
	M = len(signal)
	max_scale = int(math.floor(math.log(M) / math.log(2)))
	coeffs = pywt.wavedec(signal, 'haar', level=max_scale)
	
	smallnum = 1e-20 # To avoid zero probability

	# Wavelet energy for each scale
	Ej = np.array([wavelet_energy(coeffs[-scale])+smallnum for scale in range(1, max_scale+1)])
	E_total = np.sum(Ej)

	# Relative wavelet energy
	prob = Ej / E_total

	s1 = shannon_entropy(prob)
	s2 = math.log(max_scale) / math.log(2)
	return s1 / s2

# number of tweets containing word w , for the given interval
def N_w(tweets, w):
	count = 0.0
	for tweet in tweets:
		text = tweet['text'].lower()
		if w in text:
			count += 1.0
	return count

# Number of tweets for the given interval
def N(tweets):
	return float(len(tweets))

# Check paper to understand notation
def s_w(tweetBuckets, w, t, Tc):

	nwt = N_w(tweetBuckets[t], w)
	nt  = N(tweetBuckets[t])

	if nt == 0:
		lhs = 0
	else:
		lhs = nwt / nt

	nume = 0.0
	for i in range(0, Tc):
		nume += N(tweetBuckets[i])
	
	denom = 0.0
	for i in range(0, Tc):
		denom += N_w(tweetBuckets[i], w)


	if denom == 0:
		rhs = 0
	else:
		rhs = math.log(nume/denom)
	
	return lhs * rhs

# Check paper for noation
def second_stage(first_state_signal, delta):
	n_samples = len(first_state_signal)
	sig = []
	d_delta = 2.0 * delta
	for t in range(len(first_state_signal)//delta):
		# Add d_delta to deal with negative index#
		# And subtract 1 for array based indexing
		# This can be simplified, not to bother now
		start1 = int((t-2) * delta + 1 + d_delta - 1)
		end1 = int((t-1) * delta + d_delta -1)

		start2 = start1 + delta 
		end2 = end1 + delta

		# Range check + want full delta points
		if start1 >=0 and start2 >=0 and end1 < n_samples and end2 < n_samples:
			D_tm1 = first_state_signal[start1:end1+1]
			D_t   = first_state_signal[start2:end2+1]
			# Compute H measure over D_t
			H_tm1 = h_measure(D_tm1)
			H_t = h_measure(D_tm1+D_t)
			s_w_t = (H_t - H_tm1)/H_tm1 if H_t > H_tm1 else 0
			sig.append(s_w_t)
	return sig

def median_abs_dev(ac, median_ac):
    return np.median(np.absolute([ac[i] - median_ac for i in range(len(ac))]))

def get_words(tweetBuckets):
	words = set()
	for tweets in tweetBuckets:
		for tweet in tweets:
			tweet_tokens = preprocessing.getTokens(tweet["text"],True)
			words |= set(tweet_tokens)
	return words

def get_graph(adj, words):
	g = igraph.Graph.Adjacency(adj.astype(bool).tolist(), mode=igraph.ADJ_UNDIRECTED)
	g.vs["name"] = words
	for edge in tqdm(g.es):
		edge["weight"] = adj[edge.tuple[0]][edge.tuple[1]]
	return g

#uses fast greedy as it is the only one that is not timing out :(
def get_communities(g):
	comm = g.community_fastgreedy(weights=g.es["weight"]).as_clustering()
	for v in tqdm(g.vs):
		v["membership"] = comm.membership[v.index]
	return g, comm