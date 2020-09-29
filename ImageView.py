class ImageView:
    def __init__(self, imageId, annotationStore, statisticsStore=None):
        self.imageId = imageId
        self.annotationStore = annotationStore
        self.statisticsStore = statisticsStore
        self.checkHaveAnnotations()

    def __eq__(self, other):
        return self.imageId == other.imageId

    def __hash__(self):
        return hash(self.imageId)

    def getAnnotations(self):
        return self.annotationStore.getAnnotationSubset("image_id", self.imageId)

    def getAnnotationIndices(self):
        return self.annotationStore.getIndicesSubset("image_id", self.imageId)

    def getAnnotationIndexLabels(self):
        return self.annotationStore.getIndexLabelSubset("image_id", self.imageId)

    def getVariancePrior(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "variance_prior"]

    def getImageVariance(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "variance"]

    def getBoxVariance(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "box_variance"]

    def getExpectedNumFalseNeg(self):
        return self.statisticsStore.imageStatistics.loc[
            self.imageId, "expected_num_false_neg"
        ]

    def checkHaveAnnotations(self):
        annos = self.annotationStore.getAnnotationSubset(
            "image_id", self.imageId
        )
        self.haveAnnos = annos[~annos["empty"]].size > 0

    def haveAnnotations(self):
        return self.haveAnnos
