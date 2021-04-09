from window import Window
import numpy as np

class ParametersSelfTuning(object):

    @staticmethod
    def windowSplitting(originalWindows, newWindows):
        newWindowCutPoints = []
        for window in originalWindows:
            newWindowCutPoints.append(window.startIndex)
            newWindowCutPoints.append(window.endIndex)
        for window in newWindows:
            newWindowCutPoints.append(window.startIndex)
            newWindowCutPoints.append(window.endIndex)   
        newWindowCutPoints = list(set(newWindowCutPoints))
        newWindowCutPoints.sort()
        newWindows = []
        for window in originalWindows:
            a = newWindowCutPoints.index(window.startIndex)
            b = newWindowCutPoints.index(window.endIndex)
            splitedWindows = window.splitWindow(newWindowCutPoints[a+1:b+1])
            newWindows.extend(splitedWindows)
        for window in newWindows:
            window.clustersAdjustment()
        return newWindows

    @staticmethod
    def windowCombination(windows):
        if len(windows) == 0 or len(windows) == 1:
            return windows
        
        def combine(start):
            pre_result = []
            # print(start)
            pre_window = windows[start]
            for cluster in pre_window.clusters:
                for i in range(0, len(cluster.subsequences)):
                    i_ts_id = cluster.subsequences[i].ts_id
                    for j in range(i, len(cluster.subsequences)):
                        j_ts_id = cluster.subsequences[j].ts_id
                        if i_ts_id < j_ts_id:
                            pre_result.append((i_ts_id, j_ts_id))
                        else:
                            pre_result.append((j_ts_id, i_ts_id))


            for i in range(start+1, len(windows)):
                i_window = windows[i]
                i_result = []
                i_dict = {}
                for cluster in i_window.clusters:
                    for j in range(0, len(cluster.subsequences)):
                        i_ts_id = cluster.subsequences[j].ts_id
                        i_dict[i_ts_id] = cluster.subsequences[j]
                        for k in range(j, len(cluster.subsequences)):
                            j_ts_id = cluster.subsequences[k].ts_id
                            if i_ts_id < j_ts_id:
                                i_result.append((i_ts_id, j_ts_id))
                            else:
                                i_result.append((j_ts_id, i_ts_id))
                inter = list(set(pre_result).intersection(set(i_result)))
                uni = list(set(pre_result).union(set(i_result)))
                jc = len(inter)/len(uni)
                if jc >= 0.8:
                    pre_window.subsequences = []
                    for cluster in pre_window.clusters:
                        for subsequence in cluster.subsequences:
                            if type(subsequence.series) == np.ndarray:
                                subsequence.series = subsequence.series.tolist()
                            i_series = i_dict[subsequence.ts_id].series
                            if type(i_series) == np.ndarray:
                                i_series = i_series.tolist()
                            subsequence.series.extend(i_series)
                    pre_window.endIndex = i_window.endIndex
                    windows.remove(i_window)
                    if i > 2:
                        return i-2
                    else:
                        return 0
                else:
                    pre_result = i_result
                    pre_window = i_window
            return -1

        start = 0
        while start != -1:
            start = combine(start)

        for window in windows:
            window.computeAllDistancesAndCentroids()
            window.clustersAdjustment()

        return windows



