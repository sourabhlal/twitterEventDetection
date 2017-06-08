#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def spectral_analysis_for_dominant_period(featureTrajectory):
	"""
	Performs spectral analysis on 1 feature
	:param featureTrajectory: feature trajectory of feature being analyzed
	Returns dominantPeriod: dominant period (P_f) of feature
	Returns dominantPowerSpectrum: dominant power spectrum (S_f) of feature
	"""
	ft = np.array(featureTrajectory)
	DFT = np.fft.fft(ft)
	try:
		periodogram = np.square(np.absolute(np.split(DFT,2)[0]))
	except ValueError:
		DFT = np.delete(DFT, DFT.size-1)
		periodogram = np.square(np.absolute(np.split(DFT,2)[0]))

	dominantPeriod = len(featureTrajectory)/(np.argmax(periodogram)+1) #adding 1 because index starts from 1
	dominantPowerSpectrum = np.amax(periodogram)
	return dominantPeriod, dominantPowerSpectrum

def average_dfidf(featureTrajectory):
	return np.mean(featureTrajectory)

def heuristic_stop_word_detection(features, stopwords):
	"""
	Find max DPS, DFIDF, and min DFIDF from stopwords seed	
	:param feaures: list of features (dictionary with feature as key, feature trajectory as value)
	:param stopwords: list of stopwords
	Returns features: features without stopwords
	Returns stopwords: updated list of stopwords
	Returns UDPS: max DPS of set of stopwords
	"""
	init_trajectory = features[stopwords[0]]
	_,UDPS = spectral_analysis_for_dominant_period(init_trajectory)
	UDFIDF = average_dfidf(init_trajectory)
	LDFIDF = UDFIDF
	for sw in stopwords:
		try:
			trajectory = features[sw]
			features.pop(sw, None)
			dp,dps = spectral_analysis_for_dominant_period(trajectory)
			print (sw, dps, dp)
			avg_dfidf = average_dfidf(trajectory)
			if dps>UDPS:
				UDPS=2800  #TOFIX make this dps
			if avg_dfidf>UDFIDF:
				UDFIDF=avg_dfidf
			elif LDFIDF<avg_dfidf:
				LDFIDF=avg_dfidf
		except KeyError:
			pass
	print (UDPS)
	for f, featureTrajectory in features.items():
		_,s_f = spectral_analysis_for_dominant_period(featureTrajectory)
		average_dfidf_f = average_dfidf(featureTrajectory)
		if (s_f < UDPS) and (average_dfidf_f<=UDFIDF and average_dfidf_f>=LDFIDF):
			stopwords.append(f)
			print ("STOPWORD:", f)
	for s in stopwords:
		features.pop(s, None)
	return features, stopwords, UDPS

def categorizing_features(features):
	"""
	Categorizes features
	:param feaures: list of features (dictionary with feature as key, feature trajectory as value)
	Returns HH: set of aperiodic important events
	Returns HL: set of periodic important events
	Returns LH: set of aperiodic unimportant events
	Returns LL: set of noisy features
	"""
	HH = []
	HL = []
	LH = []
	LL = []
	SW = ["i","a","about","an","and","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","this","to","was","what","when","where","who","will","with","the","www","you", "my", "me", "so", "just", "but", "i'm"]

	features, SW, UDPS = heuristic_stop_word_detection(features, SW)
	for f, featureTrajectory in features.items():
		p_f,s_f = spectral_analysis_for_dominant_period(featureTrajectory)
		if p_f > len(featureTrajectory)/2 and s_f > UDPS:
			HH.append((f,p_f,s_f))
		elif p_f > len(featureTrajectory)/2 and s_f <= UDPS:
			LH.append((f,p_f,s_f))
		elif p_f <= len(featureTrajectory)/2 and s_f > UDPS:
			HL.append((f,p_f,s_f))
		elif p_f <= len(featureTrajectory)/2 and s_f <= UDPS:
			LL.append((f,p_f,s_f))
	return HH,LH,HL,LL