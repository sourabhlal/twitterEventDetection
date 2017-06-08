#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys, math

def KLfeatureSimilarity(fi, fj, featureTrajectories, params):
	"""
	Computes KL similarity between 2 features (max (fi->fj,fj-fi))
	:param fi: feature i
	:param fj: feature j
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	Returns: KL similarity score
	"""
	return max(KLdivergence(fi, fj, featureTrajectories, params), KLdivergence(fj, fi, featureTrajectories, params))

def KLdivergence(fi, fj, featureTrajectories, params):
	"""
	Computes KL similarity between 2 features (fi->fj)
	:param fi: feature i
	:param fj: feature j
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	Returns sum: KL similarity score
	"""
	fiTraj = featureTrajectories[fi]
	fjTraj = featureTrajectories[fj]
	sum = 0
	T = len(fiTraj)
	for  t in range(T):
		try:
			sum+= g_mix(params[fi]['weights'],params[fi]['means'],params[fi]['covariances'],fiTraj[t])*math.log((g_mix(params[fi]['weights'],params[fi]['means'],params[fi]['covariances'],fiTraj[t])/g_mix(params[fj]['weights'],params[fj]['means'],params[fj]['covariances'],fjTraj[t])))
		except ValueError:
			sum+=0
	return sum	

def g_mix(w,m,v,ft):
	return w*(1/math.sqrt(2*math.pi*(v*v)))*math.exp((-1/(2*(v*v)))*(math.pow((ft-m),2)))

def KLsetSimilarity(R, featureTrajectories, params):
	"""
	Computes KL similarity between set of features
	:param R: set of features
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	Returns: KL similarity score
	"""
	vals = []
	for fi in R:
		for fj in R:
			if fi is not fj:
				vals.append(KLfeatureSimilarity(fi[0],fj[0],featureTrajectories,params))
	return max(vals)

def featureDFOverlap(fi,fj,Mf):
	"""
	Document Frequency overlap for set of features
	:param fi: feature i
	:param fj: feature j
	:param Mf: document frequencies
	Returns: overlap score d(R)
	"""
	num = len(set(Mf[fi]) & set(Mf[fj]))
	den = min([len(x) for x in [Mf[fi],Mf[fj]]])
	return num / den

def setDFOverlap(R,Mf):
	"""
	Document Frequency overlap for set of features
	:param R: set of features
	:param Mf: document frequencies
	Returns: overlap score d(R)
	"""
	vals = []
	for fi in R:
		for fj in R:
			if fi is not fj:
				vals.append(featureDFOverlap(fi[0],fj[0],Mf))
	return min(vals)

def cost_func(Ri,featureTrajectories,params,Mf):
	"""
	Cost function for greedy event detection
	:param Ri: set of features
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	:param Mf: document frequencies
	Returns: cost C(Ri)
	"""
	num = KLsetSimilarity(Ri,featureTrajectories,params)
	sf_sum = 0
	for f in Ri:
		sf_sum+=f[2]
	den = setDFOverlap(Ri,Mf)*sf_sum
	return num/den


def argminC(Ri,sorted_HH,featureTrajectories,params,Mf):
	"""
	Find index of feature with lowest cost
	:param Ri: set of features
	:param sorted_HH: sorted list of important aperiodic events
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	:param Mf: document frequencies
	Returns: index m
	"""
	minVal = sys.maxsize
	minIdx = 0
	for i,f in enumerate(sorted_HH):
		Rif = Ri+[f]
		c = cost_func(Rif,featureTrajectories,params,Mf)
		if c<minVal:
			minVal = c
			minIdx = i
	return minIdx

def prettyPrintEvent(Ri,featureTrajectories,params):
	"""
	Returns event in readable manner
	:param Ri: set of features
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	Returns: event and its score
	"""
	ftrajs = []
	den = 0
	eventFeatures = []
	for f in Ri:
		den += f[2]
		eventFeatures.append(f[0])
	for f in Ri:
		scale = f[2]/den
		scaled_ft = [scale * x for x in featureTrajectories[f[0]]]
		ftrajs.append(scaled_ft)
	score = [sum(x) for x in zip(*ftrajs)]
	print("EVENT:", eventFeatures)
	return (eventFeatures,score)


def unsupervised_greedy_event_detection(HH,featureTrajectories,params,Mf):
	"""
	Greedy algorithm for unsupervised event detection
	:param HH: list of important aperiodic events. format: #HH = [(f,pf,sf)...]
	:param featureTrajectories: feature trajectories
	:param params: gaussian parameters
	:param Mf: document frequencies
	Returns: events found
	"""	
	events = []
	sorted_HH = list(reversed(sorted(HH, key=lambda k: k[2])))
	HH_iterator = sorted_HH[:]
	k = 0
	Ri = []
	for fi in HH_iterator:
		k+= 1
		Ri.append(fi)
		Cost_Ri = 1/fi[2]
		sorted_HH.pop(0)
		while len(sorted_HH)>0:
			m = argminC(Ri,sorted_HH,featureTrajectories,params,Mf)
			if cost_func(Ri+[sorted_HH[m]],featureTrajectories,params,Mf) < Cost_Ri:
				Ri.append(fm)
				Cost_Ri = cost_func(Ri,featureTrajectories,params,Mf)
				sorted_HH.pop(m)
			else:
				break
		events.append(prettyPrintEvent(Ri,featureTrajectories,params))
	return events