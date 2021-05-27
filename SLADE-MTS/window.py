#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
This file is contains the classes that will be used in slade-mts, including subsequences, clusters and windows
@Date     :2021/04/09 11:24:55
@Author      :XuanningHuang
@version      :1.0
'''
from grammar.segment import Segment, SegmentIndex
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy

class SubsequenceLengthNotEqual(Exception):
    def __init__(self, s1, s2):
        print('s1: ', end='')
        print(s1)
        print('s2: ', end='')
        print(s2)

class WindowSplitingNotValid(Exception): pass

class Subsequence(object):
    def __init__(self, ts_id, series = [], segments = []):
        self.segments = segments
        self.isClustered = False
        self.ts_id = ts_id
        if segments != []:
            self.series = Subsequence.segmentsConvertToSeries(segments)
        else:
            self.series = series

    @staticmethod
    def segmentsConvertToSeries(segments):
        """
        @description  :
        convert a list of the class of segments into a long time series.
        ---------
        @param  :  segments -- a list of segments
        -------
        @Returns  : timesereis
        -------
        """
        segmentsLen = len(segments)
        if segmentsLen == 0:
            return []
        seriesLen = len(segments[0].series)
        if seriesLen == 0:
            return []
        series = []
        for i in range(0, segmentsLen*seriesLen):
            segIndex = i // seriesLen
            serIndex = i % seriesLen
            series.append(segments[segIndex].series[serIndex])
        return series


    @staticmethod
    def computeSubsequenceDis(s1, s2):
        """
        @description  :
        compute the euclidean distance between two series.
        ---------
        @param  :  s1, s2 -- two seires
        -------
        @Returns  : distance(int type)
        -------
        """
        if len(s1) != len(s2):
            raise SubsequenceLengthNotEqual(s1, s2)
        sqrtSum = 0
        for i in range(0, len(s1)):
            sqrtSum += (s1[i] - s2[i])*(s1[i] - s2[i])
        return sqrtSum ** 0.5

    @staticmethod
    def computeNormalizedSubsequenceDis(s1, s2):
        """
        @description  :
        compute the normalized euclidean distance between two series.
        ---------
        @param  :  s1, s2 -- two seires
        -------
        @Returns  : normalized distance(int type)
        -------
        """
        if len(s1) != len(s2):
            raise SubsequenceLengthNotEqual()
        max1 = np.max(s1)
        min1 = np.min(s1)
        s1_n = None
        s2_n = None
        if max1 == min1:
            s1_n = []
            for i in range(0, len(s1)):
                s1_n.append(0.0)
        else:
            s1_n = (s1 - min1)/(max1 - min1)
        max2 = np.max(s2)
        min2 = np.min(s2)
        if max2 == min2:
            s2_n = []
            for i in range(0, len(s2)):
                s2_n.append(0.0)
        else:
            s2_n = (s2 - min2)/(max2 - min2)
        sqrtSum = 0
        for i in range(0, len(s1_n)):
            sqrtSum += (s1_n[i] - s2_n[i])*(s1_n[i] - s2_n[i])
        return sqrtSum ** 0.5


class Cluster(object):
    def __init__(self, subsequences):
        self.subsequences = subsequences
        self.centroid = None

    @staticmethod
    def computeCentroid(subsequences):
        """
        @description  :
        compute the centroid of a list of subsequences.
        ---------
        @param  :  subsequences -- a list of subsequences
        -------
        @Returns  : centroid -- a int array represents the centroid of the subsequences.
        -------
        """
        if subsequences == []:
            return []
        centriod = []
        subsequenceLen = len(subsequences)
        # segmentsLen = len(subsequences[0].segments)
        seriesLen = len(subsequences[0].series)
        for i in range(0, seriesLen):
            sum = 0
            for subsequence in subsequences:
                sum += subsequence.series[i]
            centriod.append(sum / subsequenceLen)
        return centriod


class Window(object):
    def __init__(self, startIndex, endIndex, segmentLength):
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.segmentLength = segmentLength
        self.subsequences = []
        self.clusters = []
        self.missedSubsequences = []
        self.distances = []
        self.distances_normalized = []
        self.threshhold = 0
        self.threshhold_normalized = 0

    def initSubsequences(self, startSeiresID, sc):
        """
        @description  :
        get the list of the subsequences from segments
        ---------
        @param  :  sc -- sc class contains all segments
        -------
        @Returns  : subsequences
        -------
        """
        self.subsequences = []
        for i in range(0, sc.tsCount):
            segments = []
            for j in range(self.startIndex, self.endIndex):
                segments.append(sc.segmentIndexes[j].segments[i])
            subsequences = Subsequence(segments = segments, ts_id = i+startSeiresID)
            self.subsequences.append(subsequences)

    def clustersCombination(self):
        """
        @description  :
        combine the clusters in the window if two of them are too similar.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        i = 0
        while i < len(self.clusters):
            j = i+1
            i_subsequences = self.clusters[i].subsequences
            changed = False
            while j < len(self.clusters):
                j_subsequences = self.clusters[j].subsequences
                interGroup = list(set(i_subsequences).intersection(set(j_subsequences)))
                similarity = max(len(interGroup)/len(i_subsequences), len(interGroup)/len(j_subsequences))
                # print('---')
                # for subsequence in i_subsequences:
                #     print('ts'+str(subsequence.ts_id), end = ' ')
                # print('\n')
                # for subsequence in j_subsequences:
                #     print('ts'+str(subsequence.ts_id), end = ' ')
                # print('\n')
                if similarity > 0.8:
                    self.clusters[i].subsequences = list(set(i_subsequences).union(set(j_subsequences)))
                    i_subsequences = self.clusters[i].subsequences
                    # for subsequence in self.clusters[i].subsequences:
                    #     print('!!!!!ts'+str(subsequence.ts_id), end = ' ')
                    # print('\n')
                    # print('---')
                    del self.clusters[j]
                    changed = True
                else:
                    j += 1
            if changed == False:
                i += 1

    def clustersBreakingTie(self):
        """
        @description  :
        break the tie of the clusters in the window if two of them are not similar.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        i = 0
        while i < len(self.clusters):
            j = i+1
            i_subsequences = self.clusters[i].subsequences
            while j < len(self.clusters):
                j_subsequences = self.clusters[j].subsequences
                interGroup = list(set(i_subsequences).intersection(set(j_subsequences)))
                i_subsequences = list(set(i_subsequences) - set(interGroup))
                j_subsequences = list(set(j_subsequences) - set(interGroup))
                if i_subsequences == []:
                    self.clusters[i].subsequences = j_subsequences
                    del self.clusters[j]
                    continue
                elif j_subsequences == []:
                    del self.clusters[j]
                    continue
                i_centroid = Cluster.computeCentroid(i_subsequences)
                j_centroid = Cluster.computeCentroid(j_subsequences)
                for subsequence in interGroup:
                    centroid = Cluster.computeCentroid([subsequence])
                    # print(centroid)
                    # print(i_centroid)
                    # print(j_centroid)
                    i_dis = Subsequence.computeSubsequenceDis(i_centroid, centroid)
                    j_dis = Subsequence.computeSubsequenceDis(j_centroid, centroid)
                    if i_dis < j_dis:
                        i_subsequences.append(subsequence)
                    else:
                        j_subsequences.append(subsequence)
                self.clusters[i].subsequences = i_subsequences
                self.clusters[j].subsequences = j_subsequences
                j += 1
            i += 1

    def clustersProcessMiss(self):
        """
        @description  :
        issue the missed subsequence to its closest cluster.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        #self.printClusters()
        if len(self.clusters) == 0:
            for subsequence in self.missedSubsequences:
                self.clusters.append(Cluster([subsequence]))
            return
        minDis = float('inf')
        minIndex = -1
        for subsequence in self.missedSubsequences:
            for i in range(0, len(self.clusters)):
                centroid = Cluster.computeCentroid(self.clusters[i].subsequences)
                dis = Subsequence.computeSubsequenceDis(subsequence.series, centroid)
                if dis < minDis:
                    minIndex = i
                    minDis = dis
            self.clusters[minIndex].subsequences.append(subsequence)
        self.missedSubsequences = []


    def initClusters(self, sc):
        """
        @description  :
        get the initial clusters from symbolic clustering class.
        ---------
        @param  : sc -- symbolic clustering class instance generated from a list of timeseries
        -------
        @Returns  :
        -------
        """
        self.clusters = []
        self.missedSubsequences = []
        rules = sc.rule_set
        for rule in rules:
            if rule == sc.grammar.root_production:
                continue
            subsequences = []
            for subsequence in self.subsequences:
                covered = True
                for segment in subsequence.segments:
                    if rule not in segment.allCoveredRules:
                        covered = False
                        break
                if covered == True:
                    subsequences.append(subsequence)
                    subsequence.isClustered = True
            if subsequences == []:
                continue
            self.clusters.append(Cluster(subsequences))
        for subsequence in self.subsequences:
            if subsequence.isClustered == False:
                self.missedSubsequences.append(subsequence)

    def updateSimilarityThreshold(self):
        """
        @description  :
        update similarity threshold of the window
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        mean = np.mean(self.distances)
        standard = np.std(self.distances)
        self.threshhold = mean + standard*3
        mean = np.mean(self.distances_normalized)
        standard = np.std(self.distances_normalized)
        self.threshhold_normalized = mean + standard*3

    def computeAllDistancesAndCentroids(self):
        """
        @description  :
        compute and update the centroids of the clusters and then compute the distance of subsequences to the centroids.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        self.ditances = []
        self.ditances_normalized = []
        for i in range(0, len(self.clusters)):
            cluster = self.clusters[i]
            centroid = Cluster.computeCentroid(cluster.subsequences)
            self.clusters[i].centroid = centroid
            for subsequence in self.clusters[i].subsequences:
                dis = Subsequence.computeSubsequenceDis(subsequence.series, centroid)
                dis_normalized = Subsequence.computeNormalizedSubsequenceDis(subsequence.series, centroid)
                self.distances.append(dis)
                self.distances_normalized.append(dis_normalized)
        self.updateSimilarityThreshold()

    def computeAllDistances(self):
        """
        @description  :
        compute the distance of subsequences to the centroids.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        self.ditances = []
        self.ditances_normalized = []
        for i in range(0, len(self.clusters)):
            cluster = self.clusters[i]
            centroid = self.clusters[i].centroid
            for subsequence in self.clusters[i].subsequences:
                dis = Subsequence.computeSubsequenceDis(subsequence.series, centroid)
                dis_normalized = Subsequence.computeNormalizedSubsequenceDis(subsequence.series, centroid)
                self.distances.append(dis)
                self.distances_normalized.append(dis_normalized)
        self.updateSimilarityThreshold()

    def splitWindow(self, splitPoints):
        """
        @description  :
        split window according to the split points.
        ---------
        @param  : splitPoints -- a list of integers
        -------
        @Returns  : new list of windows
        -------
        """
        splitPoints.sort()
        if splitPoints == [self.endIndex]:
            return [self]
        if splitPoints == [] or splitPoints[0] < self.startIndex or splitPoints[-1] > self.endIndex:
            raise WindowSplitingNotValid()
        lastPoint = self.startIndex
        newWindows = []
        for point in splitPoints:
            newWindow = Window(lastPoint, point, self.segmentLength)
            newWindow.clusters = copy.deepcopy(self.clusters)
            newWindow.subsequences = []
            for cluster in newWindow.clusters:
                for subsequence in cluster.subsequences:
                    series = subsequence.series
                    series = series[self.segmentLength*(lastPoint - self.startIndex):self.segmentLength*(point - self.startIndex)]
                    subsequence.series = series
                    newWindow.subsequences.append(subsequence)
            newWindow.computeAllDistancesAndCentroids()
            newWindows.append(newWindow)
            lastPoint = point
        return newWindows

    def clustersAdjustment(self):
        """
        @description  :
        adjust clusters after processing several timeseries by reassigning the subsequences and combining clusters.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        from _dynamic_clustering import DynamicClustering
        # Split Cluster
        for cluster in self.clusters:
            for subsequence in cluster.subsequences:
                dis = Subsequence.computeSubsequenceDis(subsequence.series, cluster.centroid)
                norm_dis = Subsequence.computeNormalizedSubsequenceDis(subsequence.series, cluster.centroid)
                #print(norm_dis)
                if dis > self.threshhold or norm_dis > self.threshhold_normalized:
                    try:
                        self.distances.remove(dis)
                        self.distances_normalized.remove(norm_dis)
                    except ValueError:
                        #print('忽略了一个错误')
                        pass
                    cluster.subsequences.remove(subsequence)
                    DynamicClustering.dynamicClusterintInWindow(subsequence, self)

        temp = self.clusters[:]
        for cluster in temp:
            if cluster.subsequences == []:
                self.clusters.remove(cluster)
        self.computeAllDistancesAndCentroids()

        # Combine Clusters
        def combineClusters():
            for i in range(0, len(self.clusters)):
                i_cluster = self.clusters[i]
                for j in range(i+1, len(self.clusters)):
                    j_cluster = self.clusters[j]
                    if Subsequence.computeSubsequenceDis(i_cluster.centroid, j_cluster.centroid) <= self.threshhold and\
                         Subsequence.computeNormalizedSubsequenceDis(i_cluster.centroid, j_cluster.centroid) <= self.threshhold_normalized:
                        i_cluster.subsequences.extend(j_cluster.subsequences) 
                        i_cluster.centroid = Cluster.computeCentroid(i_cluster.subsequences)
                        self.clusters.remove(j_cluster)
                        return True
            return False

        while combineClusters():
            pass
        
        self.computeAllDistances()

                    

    def printClusters(self):
        print('Window: '+str(self.startIndex)+'--'+str(self.endIndex))
        for i in range(0, len(self.clusters)):
            print('cluster'+str(i+1)+':', end = ' ')
            for subsequence in self.clusters[i].subsequences:
                print('ts'+str(subsequence.ts_id), end = ' ')
            print('\n')
        print('missed subsequences:', end=' ')
        for subsequence in self.missedSubsequences:
            print('ts'+str(subsequence.ts_id), end = ' ')
        print('\n\n')

    def printClustersInDetail(self):
        print('Window: '+str(self.startIndex)+'--'+str(self.endIndex))
        for i in range(0, len(self.clusters)):
            print('cluster'+str(i+1)+':', end = ' \n')
            for subsequence in self.clusters[i].subsequences:
                print('ts'+str(subsequence.ts_id), end = ' ')
                print(subsequence.series)
            print('\n')
        print('missed subsequences:', end=' ')
        for subsequence in self.missedSubsequences:
            print('ts'+str(subsequence.ts_id), end = ' ')
        print('\n\n')







