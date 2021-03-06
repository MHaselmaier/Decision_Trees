import random

from DecisionTree import DecisionTree


class RandomDecisionTree(DecisionTree):
    def __init__(self, minSamples=5, maxDepth=5):
        super().__init__(minSamples=minSamples, maxDepth=maxDepth)

    def calculatePossibleSplittingPointsPerFeature(self, featuresSortedByY):
        possibleSplittingPointsPerFeature = super().calculatePossibleSplittingPointsPerFeature(featuresSortedByY)

        featuresToExclude = random.sample(range(len(featuresSortedByY)), len(featuresSortedByY) - len(featuresSortedByY) // 3)
        for feature in featuresToExclude:
            possibleSplittingPointsPerFeature[feature] = []

        return possibleSplittingPointsPerFeature

    def prune(self, X, y):
        return
