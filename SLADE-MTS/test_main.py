#coding=utf-8
import scipy.io as sio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from _symbolic_clustering import SymbolicClustering
from _dynamic_clustering import DynamicClustering
import copy
from _parameters_self_tuning import ParametersSelfTuning
from _slade_mts import SladeMts


def mat2csv():
    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, "dataset")
    mat_data_path = os.path.join(data_path, "Etch")
    print(mat_data_path)
    if not os.path.exists(mat_data_path):
        os.makedirs(mat_data_path)
    file_list = os.listdir(mat_data_path)
    print(file_list)
    mat_list = [file_name for file_name in file_list if file_name.endswith(".mat")]

    for mat_file in mat_list:
        file_path = os.path.join(mat_data_path, mat_file)
        data = sio.loadmat(file_path)
        key = ''
        for i in data.keys():
            key = i
        print(key)
        data = data[key][0][0][1]
    return data

def TestDiscretize():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.printSegments()

def TestGrammarInduction():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        #sc.printSegments()
        sc.grammar_induction(segments)
        print(sc.grammar.print_grammar())
        print('--------------------------------------')
        #frequencyMatrix = sc.get_frequency_matrix()

def TestGetFrequencyMatrix():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
        print(sc.grammar.print_grammar())
        frequencyMatrix = sc.get_frequency_matrix()
        print(frequencyMatrix)  
        print('--------------------------------------')

def TestCutWindow():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
        frequencyMatrix = sc.get_frequency_matrix()
        windows = sc.cut_window(frequencyMatrix)
        print('--------------------------------------')
        for window in windows:
            print(str(window.startIndex)+','+str(window.endIndex))  

def TestClustersGeneration():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
    frequencyMatrix = sc.get_frequency_matrix()
    windows = sc.cut_window(frequencyMatrix)
    sc.generateInitialClusters(0, windows)
    for window in windows:
        window.printClusters()   

def TestDynamicClustering():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
    frequencyMatrix = sc.get_frequency_matrix()
    windows = sc.cut_window(frequencyMatrix)
    windows = sc.generateInitialClusters(0, windows)
    for window in windows:
        window.printClusters() 
    for i in range(10, 20):
        DynamicClustering.dynamicClustering(data[i][0][0:90, 6], windows, i, 10)
    for window in windows:
        window.printClusters() 

def TestWindowSplit():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
    frequencyMatrix = sc.get_frequency_matrix()
    windows = sc.cut_window(frequencyMatrix)
    windows = sc.generateInitialClusters(0, windows)
    print('--------0-10 by now:-------------')
    for window in windows:
        window.printClusters() 
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(10, 20):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
        DynamicClustering.dynamicClustering(data[i][0][0:90, 6], windows, i, 10)
    frequencyMatrix = sc.get_frequency_matrix()
    scwindows = sc.cut_window(frequencyMatrix)
    scwindows = sc.generateInitialClusters(10, scwindows)
    print('--------10-20 by sc:-------------')
    for window in scwindows:
        window.printClusters() 
    print('--------10-20 by now:-------------')
    for window in windows:
        window.printClusters() 
    windows = ParametersSelfTuning.windowSplitting(windows, scwindows)
    print('--------10-20 by now split:-------------')
    for window in windows:
        window.printClusters() 

def TestWindowCombination():
    data = mat2csv()
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(0, 10):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
    frequencyMatrix = sc.get_frequency_matrix()
    windows = sc.cut_window(frequencyMatrix)
    windows = sc.generateInitialClusters(0, windows)
    print('--------0-10 by now:-------------')
    for window in windows:
        window.printClusters() 
    sc = SymbolicClustering(segmentLength = 10, paaSize = 2, alphabetSize = 10, upperBound = 3000, lowerBound = 0)
    for i in range(10, 20):
        segments = sc.discretize(data[i][0][0:90, 6])
        sc.grammar_induction(segments)
        DynamicClustering.dynamicClustering(data[i][0][0:90, 6], windows, i, 10)
    frequencyMatrix = sc.get_frequency_matrix()
    scwindows = sc.cut_window(frequencyMatrix)
    scwindows = sc.generateInitialClusters(10, scwindows)
    print('--------10-20 by sc:-------------')
    for window in scwindows:
        window.printClusters() 
    print('--------10-20 by now:-------------')
    for window in windows:
        window.printClusters() 
    windows = ParametersSelfTuning.windowSplitting(windows, scwindows)
    print('--------10-20 by now split:-------------')
    for window in windows:
        window.printClusters() 
    windows = ParametersSelfTuning.windowCombination(windows)
    print('--------10-20 by now combination:-------------')
    for window in windows:
        window.printClustersInDetail() 
    print('--------0-20 original:-------------')
    for i in range(0, 20):
        print(data[i][0][0:90, 6])

def TestPatternGenerate():
    data = mat2csv()
    dataSet = []
    dataSubSet1 = []
    dataSubSet2 = []
    for i in range(0, 100):
        dataSubSet1.append(data[i][0][0:90, 6])
        dataSubSet2.append(data[i][0][0:90, 8])
    dataSet.append(dataSubSet1)
    dataSet.append(dataSubSet2)
    #print(dataSet)
    segmentLength = 10
    paaSize = 2
    alphabetSize = 10
    pStep = 10
    listBounds = []
    listBounds.append([0, 3000])
    listBounds.append([1100, 1350])
    sladeMts = SladeMts(dataSet = dataSet, listBounds = listBounds, segmentLength = segmentLength, alphabetSize = alphabetSize, paaSize = paaSize, pStep = pStep, seriesLen = 90)
    listPattern = sladeMts.patternGenerate()
    print(listPattern)

def TestDataTransformation():
    data = mat2csv()
    dataSet = []
    dataSubSet1 = []
    dataSubSet2 = []
    for i in range(0, 100):
        dataSubSet1.append(data[i][0][0:90, 6])
        dataSubSet2.append(data[i][0][0:90, 8])
    dataSet.append(dataSubSet1)
    dataSet.append(dataSubSet2)
    #print(dataSet)
    segmentLength = 10
    paaSize = 2
    alphabetSize = 10
    pStep = 10
    listBounds = []
    listBounds.append([0, 3000])
    listBounds.append([1100, 1350])
    sladeMts = SladeMts(dataSet = dataSet, listBounds = listBounds, segmentLength = segmentLength, alphabetSize = alphabetSize, paaSize = paaSize, pStep = pStep, seriesLen = 90)
    listPatterns = sladeMts.patternGenerate()
    listNewTimeSeries = sladeMts.dataTransformation(dataSet, listPatterns)
    print(listNewTimeSeries)


def TestAnomalyDetection():
    data = mat2csv()
    dataSet = []
    dataSubSet1 = []
    dataSubSet2 = []
    for i in range(0, 100):
        dataSubSet1.append(data[i][0][0:90, 6])
        dataSubSet2.append(data[i][0][0:90, 8])
    dataSet.append(dataSubSet1)
    dataSet.append(dataSubSet2)
    #print(dataSet)
    segmentLength = 10
    paaSize = 2
    alphabetSize = 10
    pStep = 10
    listBounds = []
    listBounds.append([0, 3000])
    listBounds.append([1100, 1350])
    sladeMts = SladeMts(dataSet = dataSet, listBounds = listBounds, segmentLength = segmentLength, alphabetSize = alphabetSize, paaSize = paaSize, pStep = pStep, seriesLen = 90)
    listPatterns = sladeMts.patternGenerate()
    listNewTimeSeries, patternLocations = sladeMts.dataTransformation(dataSet, listPatterns)
    # print(patternLocations)
    # print(len(patternLocations))
    # print(listNewTimeSeries)
    # print(len(listNewTimeSeries[0]))
    featureNum = len(listNewTimeSeries[0]) // 5
    selectedFeature, idx = SladeMts.featureSelection(listNewTimeSeries, featureNum)
    # print(idx)
    # print(selectedFeature)
    anomalyIndex,anomalyLocations = SladeMts.anomalyDetection_DBScan(selected_features = selectedFeature, features_idx = idx, patternLocations = patternLocations, eps = 600, min_samples = 4)
    print(anomalyIndex)
    print(anomalyLocations)



    
    
if __name__ == "__main__":
    TestDiscretize()
    TestGrammarInduction()
    TestGetFrequencyMatrix()
    TestCutWindow()
    TestClustersGeneration()
    TestDynamicClustering()
    TestWindowSplit()
    TestWindowCombination()
    TestPatternGenerate()
    TestDataTransformation()
    TestAnomalyDetection()
