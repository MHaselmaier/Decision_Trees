import random

from DecisionTree import DecisionTree


class RandomDecisionTree(DecisionTree):
    def __init__(self, minSamples=5):
        super().__init__(minSamples=5)

    def calculatePossibleSplittingPointsPerFeature(self, featuresSortedByY):
        possibleSplittingPointsPerFeature = super().calculatePossibleSplittingPointsPerFeature(featuresSortedByY)

        featuresToExclude = random.sample(range(len(featuresSortedByY)), len(featuresSortedByY) - len(featuresSortedByY) // 3)
        for feature in featuresToExclude:
            possibleSplittingPointsPerFeature[feature] = []

        return possibleSplittingPointsPerFeature
