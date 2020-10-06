import numpy as np


class WorkerView:
    def __init__(self, workerId, annotationStore, statisticsStore=None):
        self.workerId = workerId
        self.annotationStore = annotationStore
        self.statisticsStore = statisticsStore
        self.checkHaveAnnotations()

    def __eq__(self, other):
        return self.workerId == other.workerId

    def __hash__(self):
        return hash(self.workerId)

    def getAnnotations(self):
        return self.annotationStore.getAnnotationSubset("worker_id", self.workerId)

    def getAnnotationIndices(self):
        return self.annotationStore.getIndicesSubset("worker_id", self.workerId)

    def getAnnotationIndexLabels(self):
        return self.annotationStore.getIndexLabelSubset("worker_id", self.workerId)

    def getNumFalsePositive(self) -> float:
        return self.statisticsStore.workerStatistics.loc[self.workerId, "num_false_pos"]

    def getNumFalseNegative(self) -> float:
        return self.statisticsStore.workerStatistics.loc[self.workerId, "num_false_neg"]

    def getVarianceNumerator(self) -> float:
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "variance_numerator"
        ]

    def getNumFalsePositiveTrials(self) -> float:
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "num_false_pos_trials"
        ]

    def getNumFalseNegativeTrials(self) -> float:
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "num_false_neg_trials"
        ]

    def getNumVarianceTrials(self) -> float:
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "num_variance_trials"
        ]

    def getFalsePosPrior(self):
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "false_pos_prob_prior"
        ]

    def getFalseNegPrior(self):
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "false_neg_prob_prior"
        ]

    def getVariancePrior(self):
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "variance_prior"
        ]

    def getFalsePosProb(self):
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "false_pos_prob"
        ]

    def getFalseNegProb(self):
        return self.statisticsStore.workerStatistics.loc[
            self.workerId, "false_neg_prob"
        ]

    def getVariance(self):
        return self.statisticsStore.workerStatistics.loc[self.workerId, "variance"]

    def checkHaveAnnotations(self):
        annos = self.annotationStore.getAnnotationSubset("worker_id", self.workerId)
        self.haveAnnos = annos[~annos["empty"]].size > 0

    def haveAnnotations(self):
        return self.haveAnnos

    def incrementStatistics(
        self,
        numAnnos: float,
        numFalsePosTrials: float,
        numFalsePos: float,
        numFalseNegTrials: float,
        numFalseNeg: float,
        numVarianceTrials: float,
        deltaVariance: float,
    ):
        self.statisticsStore.workerStatistics.loc[
            self.workerId,
            [
                "num_annos",
                "num_false_pos_trials",
                "num_false_pos",
                "num_not_false_pos",
                "num_false_neg_trials",
                "num_false_neg",
                "num_not_false_neg",
                "num_variance_trials",
                "variance_numerator",
            ],
        ] += np.array(
            [
                numAnnos,
                numFalsePosTrials,
                numFalsePos,
                numAnnos - numFalsePos,
                numFalseNegTrials,
                numFalseNeg,
                numAnnos - numFalseNeg,
                numVarianceTrials,
                deltaVariance,
            ]
        )
