import pandas as pd
import numpy as np


class AnnotationStore:
    def __init__(self, indexCols=["image_id", "worker_id"]):
        self.indexCols = indexCols
        self.annotations = pd.DataFrame()
        self.indices = None
        self.indexGroupedAnnotations = None
        self.indexGroupedAnnotationCounts = None
        self.multiIndexGroupedAnnotations = None

    def addAnnotations(self, annotations, priors):
        self.annotations = pd.concat(
            [
                self.annotations,
                pd.concat(
                    [
                        annotations,
                        pd.DataFrame(
                            dict(
                                false_pos_prob_prior=priors["volunteer_skill"][
                                    "false_pos_prob"
                                ],
                                false_neg_prob_prior=priors["volunteer_skill"][
                                    "false_neg_prob"
                                ],
                                variance_prior=priors["volunteer_skill"]["variance"],
                                false_pos_prob=priors["volunteer_skill"][
                                    "false_pos_prob"
                                ],
                                false_neg_prob=priors["volunteer_skill"][
                                    "false_neg_prob"
                                ],
                                variance=priors["volunteer_skill"]["variance"],
                                combined_variance=priors["shared"]["variance"],
                                variance_weighting=0.5,  # balance between worker and image variance
                                association=None,
                                is_ground_truth=False,
                                matches_ground_truth=False,
                                ground_truth_distance=np.nan,
                            ),
                            index=annotations.index,
                        ),
                    ],
                    axis=1,
                ),
            ]
        )

        self.indices = {
            indexCol: self.annotations.groupby(by=indexCol).indices
            for indexCol in self.indexCols
        }

    def getAnnotations(self):
        return self.annotations

    def getAnnotationsArray(self):
        return self.annotations.values

    def getAnnotationSubset(self, indexCol=None, indexKey=None):
        return self.annotations.loc[self.indices[indexCol][indexKey]]

    def getIndicesSubset(self, indexCol=None, indexKey=None):
        return self.indices[indexCol][indexKey]

    def getAnnotationsSubsetArray(self, indexCol, indexKey):
        return self.getAnnotations(indexCol, indexKey).values

    def getIndexValues(self, indexCol):
        return self.indices[indexCol].keys()

    def getIndexValuesArray(self, indexCol):
        return np.fromiter(self.getIndexValues(indexCol), dtype=np.int64)

    def generateViews(self, indexCol, viewType, statStore):
        return [
            viewType(typeId, self, statStore)
            for typeId in self.getIndexValues(indexCol)
        ]

    def getNumAnnotations(self):
        return self.annotations.shape[0]

    def rebuildIndexGroups(self):
        self.indexGroupedAnnotations = {
            indexCol: self.annotations.groupby(by=indexCol)
            for indexCol in self.indexCols
        }
        self.indexGroupedAnnotationCounts = {
            indexCol: groupedAnnotations[indexCol].count()
            for indexCol, groupedAnnotations in self.indexGroupedAnnotations.items()
        }

    def rebuildMultiIndexGroups(self):
        self.multiIndexGroupedAnnotations = self.annotations.groupby(by=self.indexCols)

    def getIndexGroupedAnnotationCounts(self, indexCol, rebuild=False):
        if rebuild or self.indexGroupedAnnotations is None:
            self.rebuildIndexGroups()
        return self.indexGroupedAnnotationCounts[indexCol]

    def getIndexGroupedAnnotations(self, indexCol, rebuild=False):
        if rebuild or self.indexGroupedAnnotationCounts is None:
            self.rebuildIndexGroups()
        return self.indexGroupedAnnotations[indexCol]

    def getMultiIndexGroupedAnnotations(self, rebuild=False):
        if rebuild or self.multiIndexGroupedAnnotations is None:
            self.rebuildMultiIndexGroups()
        return self.multiIndexGroupedAnnotations
