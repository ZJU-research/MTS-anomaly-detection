#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
This file is used for the multivariate time series anomaly detection, based on
the MCFS feature selection and the DBScan clustering method. 
@Date     :2021/04/09 11:24:55
@Author      :XuanningHuang
@version      :1.0
'''

from _dynamic_clustering import DynamicClustering
from _parameters_self_tuning import ParametersSelfTuning
from _symbolic_clustering import SymbolicClustering
from window import Window, Subsequence, Cluster
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W
import numpy as np
import dbscan
from sklearn.cluster import DBSCAN
import copy


class PStepIsNotValid(Exception): pass

class SladeMts(object):

    def __init__(self, dataSet, listBounds, segmentLength, alphabetSize, paaSize, pStep, seriesLen):
        self.dataSet = dataSet
        self.listBounds = listBounds
        self.segmentLength = segmentLength
        self.alphabetSize = alphabetSize
        self.paaSize = paaSize
        self.pStep = pStep
        self.seriesLen = seriesLen
        

    def patternGenerate(self):
        """
        @description  :
        generate patterns from the timeseries, where we will use the symbolic clustering method 
        to generate windows in each dimension. The pattern is defined as the center of clusters 
        from each window.
        ---------
        @param  :
        -------
        @Returns  : listPatterns
        -------
        """
        
        
        print('正在生成patterns...')
        numDim = len(self.dataSet)
        listPatterns = []
        for i in range(0, numDim):
            print(i / numDim)
            timeSeries = self.dataSet[i]
            seriesNum = len(timeSeries)
            if self.pStep > seriesNum:
                raise PStepIsNotValid()
            sc = SymbolicClustering(segmentLength = self.segmentLength, paaSize = self.paaSize, alphabetSize = self.alphabetSize, upperBound = self.listBounds[i][1], lowerBound = self.listBounds[i][0])
            for j in range(0, self.pStep):
                # print(timeSeries[j])
                segments = sc.discretize(timeSeries[j])
                sc.grammar_induction(segments)
            frequencyMatrix = sc.get_frequency_matrix()
            windows = sc.cut_window(frequencyMatrix)
            windows = sc.generateInitialClusters(0, windows)
            # for window in windows:
            #     window.printClustersInDetail()
            dynamicStep = self.pStep
            sc = SymbolicClustering(segmentLength = self.segmentLength, paaSize = self.paaSize, alphabetSize = self.alphabetSize, upperBound = self.listBounds[i][1], lowerBound = self.listBounds[i][0])
            while dynamicStep < seriesNum:
                if len(timeSeries[dynamicStep]) < self.seriesLen:
                    # for window in windows:
                    #     window.printClustersInDetail() 
                    timeSeries[dynamicStep] = timeSeries[dynamicStep].tolist()
                    for j in range(len(timeSeries[dynamicStep]), self.seriesLen):
                        timeSeries[dynamicStep].append(0.0)
                    print(timeSeries[dynamicStep])


                DynamicClustering.dynamicClustering(timeSeries[dynamicStep], windows, dynamicStep, self.segmentLength)
                segments = sc.discretize(timeSeries[dynamicStep])
                sc.grammar_induction(segments)
                if (dynamicStep + 1) % self.pStep == 0:
                    frequencyMatrix = sc.get_frequency_matrix()
                    scwindows = sc.cut_window(frequencyMatrix)
                    # scwindows = sc.generateInitialClusters(dynamicStep + 1 - self.pStep, scwindows)
                    windows = ParametersSelfTuning.windowSplitting(windows, scwindows)
                    windows = ParametersSelfTuning.windowCombination(windows)
                    sc = SymbolicClustering(segmentLength = self.segmentLength, paaSize = self.paaSize, alphabetSize = self.alphabetSize, upperBound = self.listBounds[i][1], lowerBound = self.listBounds[i][0])
                dynamicStep += 1

        
            listPatterns.append(windows)
        return listPatterns

    def dataTransformation(self, dataSet, listPatterns):
        """
        @description  : transfrom the timeseries to a set of features
        ---------
        @param  : 
        -------
        @Returns  : listNewTimeSeries -- timeseries in new feature space
                    patternLocations -- the location of the patterns that produce the features respectively
        -------
        """
                
        numDim = len(self.dataSet)
        if numDim == 0:
            return
        numSeries = len(self.dataSet[0])
        listNewTimeSeries = []
        patternLocations = []
        first = True
        for i in range(0, numSeries):
            print(i)
            transformed = []
            for j in range(0, numDim):
                s = dataSet[j][i]
                if len(s) < self.seriesLen:
                    s = s.tolist()
                    for k in range(len(s), self.seriesLen):
                        s.append(0.0)
                windows = listPatterns[j]
                for window in windows:
                    ss = window.startIndex * self.segmentLength
                    ee = window.endIndex * self.segmentLength
                    subseries = s[ss:ee]
                    for cluster in window.clusters:
                        dis = Subsequence.computeSubsequenceDis(subseries, cluster.centroid)
                        transformed.append(dis)
                        if first == True:
                            patternLocations.append((j, ss, ee))

            first = False

            listNewTimeSeries.append(transformed)
        
        return listNewTimeSeries, patternLocations

    @staticmethod
    def featureSelection(listNewTimeSeries, numFeature, numCluster = 5):
        """
        @description  : select features using MCFS algorithem.
        ---------
        @param  : numFeature -- how many features are to be selected.
                  numCluster -- parameter required in MFCS, deafault is set to 5.
        -------
        @Returns  : selected_features -- selected features
                    idx -- the indexes of selected features in original feature set.
        -------
        """
        
        
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        listNewTimeSeries = np.array(listNewTimeSeries)
        W = construct_W.construct_W(listNewTimeSeries, **kwargs)
        Weight = MCFS.mcfs(listNewTimeSeries, n_selected_features=numFeature, W=W, n_clusters=numCluster)
        idx = MCFS.feature_ranking(Weight)
        selected_features = listNewTimeSeries[:, idx[0:numFeature]]
        return selected_features, idx
    
    @staticmethod
    def DBScanParametersChoose(selected_features):
        """
        @description  : this function is used to test parameters that is going to be adapted in DBScan.
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
        res = []
        # 迭代不同的eps值
        EpsCandidate = dbscan.returnEpsCandidate(selected_features)
        for eps in EpsCandidate:
            # 迭代不同的min_samples值
            for min_samples in range(2,10):
                dbscan = DBSCAN(eps = eps, min_samples = min_samples)
                # 模型拟合
                dbscan.fit(selected_features)
                # 统计各参数组合下的聚类个数（-1表示异常点）
                n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
                # 异常点的个数
                outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
                # 统计每个簇的样本个数
                stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
                res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
        # 将迭代后的结果存储到数据框中        
        df = pd.DataFrame(res)
        return df


    @staticmethod
    def anomalyDetection_DBScan(selected_features, features_idx, patternLocations, eps = 600, min_samples = 4, sensibility = 3):
        """
        @description  : detect the anomaly series and know the locations of them.
        ---------
        @param  : eps, min_sample: parameter required in DBScan.
                  features_idx, patternLocations: parameters that is used to know the feature locations.
                  sensibility: used to define the threshold of anomaly with the anomaly scores.
        -------
        @Returns  : anomalyIndex: the anomaly series number.
                    anomalyLocations: the anomaly features locations in each anomaly series.
        -------
        """
        
        
        clustering = DBSCAN(eps = eps,min_samples = min_samples).fit(selected_features)
        labels = clustering.labels_
        numCluster = max(clustering.labels_)+1
        clusters = []
        centroids = []
        tSupport = -1
        cBig = -1
        for i in range(0, numCluster):
            cluster = [j for j, x in enumerate(labels) if x == i]
            centriod = []
            clusterLen = len(cluster)
            if clusterLen > cBig:
                cBig = clusterLen
                tSupport = clusterLen // 2
            seriesLen = len(selected_features[0])
            for j in range(0, seriesLen):
                sum = 0
                for k in cluster:
                    sum += selected_features[k][j]
                centriod.append(sum / clusterLen)
            centroids.append(centriod)
            clusters.append(cluster)

        enough = []
        enoughTags = []
        for i in range(0, numCluster):
            clusterLen = len(clusters[i])
            if clusterLen >= tSupport:
                enough.append(i)
                enoughTags.append(True)
            else:
                enoughTags.append(False)
        
        anomalyScores = []
        closestIndexes = []
        for i in range(0, len(labels)):
            closestDis = float('inf')
            closestIndex = -1
            if labels[i] == -1 or enoughTags[labels[i]] == False:
                for index in enough:
                    dis = Subsequence.computeSubsequenceDis(selected_features[i], centroids[index])
                    if dis < closestDis:
                        closestDis = dis
                        closestIndex = index
            else:
                closestIndex = labels[i]
                closestDis = Subsequence.computeSubsequenceDis(selected_features[i], centroids[closestIndex])
            
            closestIndexes.append(closestIndex)

            if labels[i] == -1:
                anomalyScore = closestDis*(1-1/cBig)
            else:
                anomalyScore = closestDis*(1-len(clusters[labels[i]])/cBig)

            anomalyScores.append(anomalyScore)

        mean = np.mean(anomalyScores)
        standard = np.std(anomalyScores)
        threshhold = mean + standard*sensibility
        anomalyIndex = []
        for i in range(0, len(anomalyScores)):
            if anomalyScores[i] >= threshhold:
                anomalyIndex.append(i)

        normalClusters = copy.deepcopy(clusters)
        for anomaly in anomalyIndex:
            for cluster in normalClusters:
                if anomaly in cluster:
                    cluster.remove(anomaly)
        
        anomalyLocations = []
        for anomaly in anomalyIndex:
            anomalyLocation = []
            normalClusterIndex = -1
            if labels[anomaly] == -1 or normalClusters[labels[anomaly]] == []:
                normalClusterIndex = closestIndexes[anomaly]
            else:
                normalClusterIndex = labels[anomaly]

            for i in range(0, len(selected_features[0])):
                features = []
                for index in normalClusters[normalClusterIndex]:
                    features.append(selected_features[index][i])
                mean = np.mean(features)
                standard = np.std(features)
                if abs(selected_features[anomaly][i] - mean) > sensibility*standard:
                    anomalyLocation.append(patternLocations[features_idx[i]])
                
            anomalyLocation = list(set(anomalyLocation))
            anomalyLocations.append(anomalyLocation)

        return anomalyIndex, anomalyLocations












    