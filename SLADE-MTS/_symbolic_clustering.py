#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
This file is the realization of symbolic clustering algorithem
@Date     :2021/04/09 11:56:04
@Author      :XuanningHuang
@version      :1.0
'''

from saxpy import SAX
import numpy as np
from grammar.grammar import Grammar
from grammar.segment import Segment, SegmentIndex
from window import Window

class SegmentsCanNotBeEquallyDivided(Exception): pass

class SymbolicClustering(object):

    def __init__(self, segmentLength = 20, paaSize = 5, alphabetSize = 3, upperBound = 100, lowerBound = -100):
        self.segmentLength = segmentLength
        self.paaSize = paaSize
        self.alphabetSize = alphabetSize
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.sax = SAX(wordSize = paaSize, alphabetSize = alphabetSize, lowerBound = lowerBound, upperBound = upperBound, epsilon = 1e-6)
        self.grammar = Grammar()
        self.segmentIndexes = []
        self.rule_set = []
        self.tsCount = 0

    def printSegments(self):
        print("\nCurrent Segments:")
        for segmentIndex in self.segmentIndexes:
            segmentIndex.printContent()

    def discretize(self, s):
        """
        @description  : discretize the single seires using modified PAA method.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
        n = len(s)
        segments = []
        if n % self.segmentLength != 0:
           raise SegmentsCanNotBeEquallyDivided() 
        nSegment = int(n / self.segmentLength)
        for i in range(0, nSegment):
            start = i*self.segmentLength
            end = (i+1)*self.segmentLength
            if self.tsCount == 0:
                self.segmentIndexes.append(SegmentIndex((start, end)))
            (letters, indices) = self.sax.to_letter_rep_ori(s[start:end])
            segment = Segment(s[start:end], letters, indices, self.segmentIndexes[i])
            self.segmentIndexes[i].addSegment(segment)
            segments.append(segment)
        self.tsCount += 1
        return segments
    

    def grammar_induction(self, segments):
        self.grammar.train_string(segments)
        self.rule_set = self.grammar.get_rule_set()

    def get_frequency_matrix(self):
        """
        @description  : for each segment, generate their frequencies of which are covered by the same grammar rule.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
        frequencyMatrix = []
        for segmentIndex in self.segmentIndexes:
            rDict = {}
            for j in range(0, self.tsCount):
                segment = segmentIndex.getSegment(j)
                rule = segment.getRule()
                if rDict.get(rule) == None:
                    rDict[rule] = 1
                else:
                    rDict[rule] = rDict[rule] + 1

            rowFrequency = []
            for j in range(0, self.tsCount):
                rule = segmentIndex.getSegment(j).getRule()
                if rule == self.grammar.root_production:
                    rowFrequency.append(1)
                else:
                    rowFrequency.append(rDict[segmentIndex.getSegment(j).getRule()])

            frequencyMatrix.append(rowFrequency)
        
        return frequencyMatrix

    def cut_window(self, frequencyMatrix):
        """
        @description  : generate windows with the frequencyMatrix. The change points of the matrix are the cut lines.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
        start = 0
        windows = []
        for now in range(1, len(frequencyMatrix)):
            if frequencyMatrix[now] != frequencyMatrix[start]:
                windows.append(Window(start, now, self.segmentLength))
                start = now
        windows.append(Window(start, len(frequencyMatrix), self.segmentLength))
        return windows

    def generateInitialClusters(self, startIndex, windows):
        """
        @description  : generate initial clusters in each window. The clusters are not 
        overlapped by each other but the sum of them covers all the segments. 
        ---------
        @param  : startIndex -- the start number of p time series.
        -------
        @Returns  :
        -------
        """
        
        
        for window in windows:
            window.initSubsequences(startIndex, self)
            window.initClusters(self)
            window.clustersCombination()
            window.clustersBreakingTie()
            window.clustersProcessMiss()
            window.computeAllDistancesAndCentroids()
        return windows
            






            
 



