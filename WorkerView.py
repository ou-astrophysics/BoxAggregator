class WorkerView:
    def __init__(self, workerId, annotationStore, statisticsStore=None):
        self.workerId = workerId
        self.annotationStore = annotationStore
        self.statisticsStore = statisticsStore

    def getAnnotations(self):
        return self.annotationStore.getAnnotationSubset("worker_id", self.workerId)

    def getAnnotationIndices(self):
        return self.annotationStore.getIndicesSubset("worker_id", self.workerId)

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