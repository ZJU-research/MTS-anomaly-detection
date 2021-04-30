import copy
import random
import numpy as np
from dataProcess import *
from parameters import *

FLOAT_MAX = 1e100


def solveDistanceBetweenPoints(pointA, pointB, lam):
    dis = 0.0
    lam_sum = 0.0
    row = pointA.shape[0]
    for i in range(row - 1):
        dis += lam[i] * np.linalg.norm(pointA[i, :] - pointB[i, :], 2)
        lam_sum += lam[i]

    dis += (1 - lam_sum) * np.linalg.norm(pointA[row - 1, :] - pointB[row - 1, :], 2)
    return dis


def getNearestCenter(subseq, clusterCenterGroup, lam):
    minIndex = subseq.group
    minDistance = FLOAT_MAX
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(subseq.mat, center.mat, lam)
        if (distance < minDistance):
            minDistance = distance
            minIndex = index
    return (minIndex, minDistance)


def kMeansPlusPlus(subsequences, clusterCenterGroup, lam):
    clusterCenterGroup[0] = copy.copy(random.choice(subsequences))
    distanceGroup = [0.0 for _ in range(len(subsequences))]
    sum = 0.0
    for index in range(1, len(clusterCenterGroup)):
        for i, subseq in enumerate(subsequences):
            distanceGroup[i] = getNearestCenter(subseq, clusterCenterGroup[:index], lam)[1]
            sum += distanceGroup[i]
        sum *= random.random()
        for i, distance in enumerate(distanceGroup):
            sum -= distance
            if sum < 0:
                clusterCenterGroup[index] = copy.copy(subsequences[i])
                break
    return


def fuzzyCMeansClustering(lam):
    subsequences = data_processing()
    print("----------------FCM----------------")
    clusterCenterGroup = [Subsequence() for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(subsequences, clusterCenterGroup, lam)
    # clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
    currentObjective = FLOAT_MAX
    newObjective = 0.0
    while True:
        print("loop")
        for subseq in subsequences:
            getSingleMembership(subseq, clusterCenterGroup, weight, lam)
        currentCenterGroup = [Subsequence() for _ in range(clusterCenterNumber)]
        for centerIndex, center in enumerate(currentCenterGroup):
            upperSum, lowerSum = 0.0, 0.0
            for subseq in subsequences:
                membershipWeight = pow(subseq.membership[centerIndex], weight)
                upperSum += subseq.mat * membershipWeight
                lowerSum += membershipWeight
            center.mat = upperSum / lowerSum
        newObjective = 0.0
        for centerIndex, center in enumerate(currentCenterGroup):
            for subseq in subsequences:
                membershipWeight = pow(subseq.membership[centerIndex], weight)
                newObjective += membershipWeight * solveDistanceBetweenPoints(subseq.mat, center.mat, lam)
        clusterCenterGroup = currentCenterGroup
        print("currentObjective:", currentObjective)
        print("newObjective:", newObjective)
        if currentObjective - newObjective <= 1e-5:
            break
        currentObjective = newObjective

    subsequences_hat = reconstruction(subsequences, clusterCenterGroup)
    reconstructionError = 0.0
    for index in range(N):
        reconstructionError += solveDistanceBetweenPoints(subsequences[index].mat, subsequences_hat[index].mat, lam)

    return reconstructionError


def getSingleMembership(point, clusterCenterGroup, weight, lam):
    distanceFromPoint2ClusterCenterGroup = [solveDistanceBetweenPoints(point.mat, clusterCenterGroup[index].mat, lam)
                                            for index in range(len(clusterCenterGroup))]
    for centerIndex, singleMembership in enumerate(point.membership):
        sum = 0.0
        isCoincide = [False, 0]
        for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
            if distance == 0:
                isCoincide[0] = True
                isCoincide[1] = index
                break
            sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance), 2.0 / (weight - 1.0))
        if isCoincide[0]:
            if isCoincide[1] == centerIndex:
                point.membership[centerIndex] = 1.0
            else:
                point.membership[centerIndex] = 0.0
        else:
            point.membership[centerIndex] = 1.0 / sum


# reconstruction
def reconstruction(subsequences, clusterCenterGroup):
    subsequences_hat = [Subsequence() for _ in range(N)]
    for index, subseq_hat in enumerate(subsequences_hat):
        uppersum = 0.0
        lowerSum = 0.0
        for centerIndex, center in enumerate(clusterCenterGroup):
            membershipWeight = pow(subsequences[index].membership[centerIndex], weight)
            uppersum += membershipWeight * center.mat
            lowerSum += membershipWeight
        subseq_hat.mat = uppersum / lowerSum
    return subsequences_hat




import copy
import random
import numpy as np
from dataProcess import *
from parameters import *

FLOAT_MAX = 1e100


class fuzzyCmeansClustering:
    def __init__(self, q, clusterCenterNumber):
        N = int((p - q) / r) + 1
        self.subsequences = data_processing(q, N, clusterCenterNumber)
        self.subsequences_hat = [Subsequence(q, clusterCenterNumber) for _ in range(N)]
        self.anomaly_score_per_win = [0.0 for _ in range(N)]
        self.lam = None
        self.q = q
        self.clusterCenterNumber = clusterCenterNumber

    def solveDistanceBetweenPoints(self, pointA, pointB, lam):
        dis = 0.0
        lam_sum = 0.0
        row = pointA.shape[0]
        for i in range(row - 1):
            dis += lam[i] * np.linalg.norm(pointA[i, :] - pointB[i, :], 2)
            lam_sum += lam[i]

        dis += (1 - lam_sum) * np.linalg.norm(pointA[row - 1, :] - pointB[row - 1, :], 2)
        return dis

    def getNearestCenter(self, subseq, clusterCenterGroup, lam):
        minIndex = subseq.group
        minDistance = FLOAT_MAX
        for index, center in enumerate(clusterCenterGroup):
            distance = self.solveDistanceBetweenPoints(subseq.mat, center.mat, lam)
            if (distance < minDistance):
                minDistance = distance
                minIndex = index
        return (minIndex, minDistance)

    def kMeansPlusPlus(self, subsequences, clusterCenterGroup, lam):
        clusterCenterGroup[0] = copy.copy(random.choice(subsequences))
        distanceGroup = [0.0 for _ in range(len(subsequences))]
        sum = 0.0
        for index in range(1, len(clusterCenterGroup)):
            for i, subseq in enumerate(subsequences):
                distanceGroup[i] = self.getNearestCenter(subseq, clusterCenterGroup[:index], lam)[1]
                sum += distanceGroup[i]
            sum *= random.random()
            for i, distance in enumerate(distanceGroup):
                sum -= distance
                if sum < 0:
                    clusterCenterGroup[index] = copy.copy(subsequences[i])
                    break
        return

    def fuzzyCMeansClustering(self, lam):
        print("----------------FCM----------------")
        self.lam = lam
        clusterCenterGroup = [Subsequence(self.q, self.clusterCenterNumber) for _ in range(self.clusterCenterNumber)]
        self.kMeansPlusPlus(self.subsequences, clusterCenterGroup, lam)
        # clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
        currentObjective = FLOAT_MAX
        newObjective = 0.0
        while True:
            print("loop")
            for subseq in self.subsequences:
                self.getSingleMembership(subseq, clusterCenterGroup, weight, lam)
            currentCenterGroup = [Subsequence() for _ in range(self.clusterCenterNumber)]
            for centerIndex, center in enumerate(currentCenterGroup):
                upperSum, lowerSum = 0.0, 0.0
                for subseq in self.subsequences:
                    membershipWeight = pow(subseq.membership[centerIndex], weight)
                    upperSum += subseq.mat * membershipWeight
                    lowerSum += membershipWeight
                center.mat = upperSum / lowerSum
            newObjective = 0.0
            for centerIndex, center in enumerate(currentCenterGroup):
                for subseq in self.subsequences:
                    membershipWeight = pow(subseq.membership[centerIndex], weight)
                    newObjective += membershipWeight * self.solveDistanceBetweenPoints(subseq.mat, center.mat, lam)
            clusterCenterGroup = currentCenterGroup
            print("currentObjective:", currentObjective)
            print("newObjective:", newObjective)
            if currentObjective - newObjective <= 1e-1:
                break
            currentObjective = newObjective

        self.reconstruction(self.subsequences, clusterCenterGroup)
        reconstructionError = self.cal_anomaly_score_per_win()

        return reconstructionError

    def getSingleMembership(self, point, clusterCenterGroup, weight, lam):
        distanceFromPoint2ClusterCenterGroup = [
            self.solveDistanceBetweenPoints(point.mat, clusterCenterGroup[index].mat, lam)
            for index in range(len(clusterCenterGroup))]
        for centerIndex, singleMembership in enumerate(point.membership):
            sum = 0.0
            isCoincide = [False, 0]
            for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
                if distance == 0:
                    isCoincide[0] = True
                    isCoincide[1] = index
                    break
                sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance), 2.0 / (weight - 1.0))
            if isCoincide[0]:
                if isCoincide[1] == centerIndex:
                    point.membership[centerIndex] = 1.0
                else:
                    point.membership[centerIndex] = 0.0
            else:
                point.membership[centerIndex] = 1.0 / sum

    # reconstruction
    def reconstruction(self, subsequences, clusterCenterGroup):
        for index, subseq_hat in enumerate(self.subsequences_hat):
            uppersum = 0.0
            lowerSum = 0.0
            for centerIndex, center in enumerate(clusterCenterGroup):
                membershipWeight = pow(subsequences[index].membership[centerIndex], weight)
                uppersum += membershipWeight * center.mat
                lowerSum += membershipWeight
            subseq_hat.mat = uppersum / lowerSum
        return

    def cal_anomaly_score_per_win(self):
        reconstructionError = 0.0
        for index in range(N):
            self.anomaly_score_per_win[index] = self.solveDistanceBetweenPoints(
                self.subsequences[index].mat, self.subsequences_hat[index].mat, self.lam)
            reconstructionError += self.anomaly_score_per_win[index]
            return reconstructionError

    def cal_anomaly(self):
        anomaly_score = np.zeros(p)
        for i in range(p):
            if i < self.q:
                anomaly_score[i] = sum([(self.anomaly_score_per_win[j]) for j in range(0, i + 1)]) / (i + 1)
            elif self.q <= i <= p - self.q:
                anomaly_score[i] = sum(
                    [(self.anomaly_score_per_win[j]) for j in range(i - self.q + 1, i + 1)]) / self.q
            else:
                anomaly_score[i] = sum(
                    [(self.anomaly_score_per_win[j]) for j in range(i - self.q + 1, p - self.q + 1)]) / (p - i)
        return anomaly_score
