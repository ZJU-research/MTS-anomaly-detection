#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
This file contains the method of clustering the new series in stream with existed windows.
@Date     :2021/04/09 12:07:15
@Author      :XuanningHuang
@version      :1.0
'''

from window import Window, Subsequence, Cluster

class SeriesIsNotLongEnough(Exception): pass

class DynamicClustering(object):

    @staticmethod
    def dynamicClustering(s, windows, ts_id, segmentLength):
        """
        @description  :
        when a series come, cut it to windows and assign each subseries to closest cluster.
        ---------
        @param  : s -- coming series
                  windows -- current windows with clustered timeseries
                  ts_id -- the number of coming timeseries
                  segmentLength -- the length of segment set before
        -------
        @Returns  :
        -------
        """
        
        
        s_len = len(s)
        for window in windows:
            d_min = float('inf')
            d_norm_min = float('inf')
            minIndex = -1
            if s_len < window.endIndex * segmentLength:
                raise SeriesIsNotLongEnough()
            subseries = s[window.startIndex * segmentLength:window.endIndex * segmentLength]
            for i in range(0, len(window.clusters)):
                cluster = window.clusters[i]
                # print(window.threshhold)
                # print(window.threshhold_normalized)
                dis = Subsequence.computeSubsequenceDis(subseries, cluster.centroid)
                norm_dis = Subsequence.computeNormalizedSubsequenceDis(subseries, cluster.centroid)
                if dis <= window.threshhold and norm_dis <= window.threshhold_normalized and dis < d_min:
                    d_min = dis
                    d_norm_min = norm_dis
                    minIndex = i
            if minIndex == -1:
                cluster = Cluster([Subsequence(ts_id = ts_id, series = subseries)])
                cluster.centroid = subseries
                window.clusters.append(cluster)
            else:
                window.clusters[minIndex].subsequences.append(Subsequence(ts_id = ts_id, series = subseries))
                window.distances.append(d_min)
                window.ditances_normalized.append(d_norm_min)
                window.updateSimilarityThreshold()

    @staticmethod
    def dynamicClusterintInWindow(subsequence, window):
        """
        @description  : cluster a subsequence in a given window.
        ---------
        @param  : subsequence -- a sub part of a series, only belongs to one certain window
                  window -- the window that the subsequence will be clustered in
        -------
        @Returns  :
        -------
        """
        
        
        d_min = float('inf')
        d_norm_min = float('inf')
        minIndex = -1
        subseries = subsequence.series
        for i in range(0, len(window.clusters)):
            cluster = window.clusters[i]
            # print(window.threshhold)
            # print(window.threshhold_normalized)
            dis = Subsequence.computeSubsequenceDis(subseries, cluster.centroid)
            norm_dis = Subsequence.computeNormalizedSubsequenceDis(subseries, cluster.centroid)
            if dis <= window.threshhold and norm_dis <= window.threshhold_normalized and dis < d_min:
                d_min = dis
                d_norm_min = norm_dis
                minIndex = i
        if minIndex == -1:
            cluster = Cluster([subsequence])
            cluster.centroid = subseries
            window.clusters.append(cluster)
        else:
            window.clusters[minIndex].subsequences.append(subsequence)
            window.distances.append(d_min)
            window.ditances_normalized.append(d_norm_min)
            window.updateSimilarityThreshold()

    






        
