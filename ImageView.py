class ImageView:
    def __init__(self, imageId, annotationStore, statisticsStore=None):
        self.imageId = imageId
        self.annotationStore = annotationStore
        self.statisticsStore = statisticsStore

    def getAnnotations(self):
        return self.annotationStore.getAnnotationSubset("image_id", self.imageId)

    def getAnnotationIndices(self):
        return self.annotationStore.getIndicesSubset("image_id", self.imageId)

    def getVariancePrior(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "variance_prior"]
    
    def getImageVariance(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "variance"]
    
    def getBoxVariance(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "box_variance"]
    
    def getExpectedNumFalseNeg(self):
        return self.statisticsStore.imageStatistics.loc[self.imageId, "expected_num_false_neg"]
    