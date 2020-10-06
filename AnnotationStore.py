import pandas as pd
import numpy as np

from SaveableStore import SaveableStore
from StatisticStore import StatisticStore


class AnnotationStore(SaveableStore):

    saveableAttrs = [
        "indexCols",
        "annotations",
        "indices",
        "indexGroupedAnnotations",
        "indexGroupedAnnotationCounts",
        "multiIndexGroupedAnnotations",
    ]

    def __init__(self, indexCols=["image_id", "worker_id"]):
        self.indexCols = indexCols
        self.annotations = pd.DataFrame()
        self.indices = None
        self.indexGroupedAnnotations = None
        self.indexGroupedAnnotationCounts = None
        self.multiIndexGroupedAnnotations = None

    def addAnnotations(
        self, annotations: pd.DataFrame, priors: dict, statisticStore: StatisticStore
    ) -> None:
        """Add new annotations to the store. Optionally filter images that are
        already finished.

        Parameters
        ----------
        annotations : pandas.DataFrame
            A dataframe containing the data for a new batch of annotations.
        priors : dict
            Dataset-wide prior values used to initialise values for new
            annotations.
        statisticStore : StatisticStore
            Used to determine whether any annotations apply to images that
            are already finished and determine skill parameters for workers who
            contributed to earlier batches.

        Returns
        -------
        None
            Description of returned object.

        """
        finished = pd.Index([])
        if statisticStore.imageStatistics.index.size > 0:
            finished = statisticStore.imageStatistics.loc[
                :, statisticStore.imageStatistics.is_finished
            ].index
            annotations = annotations.groupby(by="image_id").filter(
                lambda grp: grp.name not in finished
            )

        # Set worker skills for relevant annotations
        if statisticStore.workerStatistics.index.size > 0:
            skillParameterColumns = ["variance", "false_pos_prob", "false_neg_prob"]
            annotations = annotations.copy(deep=True).set_index("worker_id")
            annotations.loc[
                statisticStore.workerStatistics.index, skillParameterColumns
            ] = statisticStore.workerStatistics[skillParameterColumns]

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
                                connection_cost=np.nan,
                                is_ground_truth=False,
                                is_cleanup_ground_truth=False,
                                is_pruned=False,
                                is_merged=False,
                                prune_distance=np.nan,
                                matches_ground_truth=False,
                                ground_truth_distance=np.nan,
                                multiplicity_weights=1,
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
        return self.annotations.iloc[self.indices[indexCol][indexKey]]

    def getIndicesSubset(self, indexCol=None, indexKey=None):
        return self.indices[indexCol][indexKey]

    def getIndexLabelSubset(self, indexCol=None, indexKey=None):
        return self.annotations.index[self.getIndicesSubset(indexCol, indexKey)]

    def getAnnotationsSubsetArray(self, indexCol, indexKey):
        return self.getAnnotationSubset(indexCol, indexKey).values

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

    def setConnectionCosts(self, connectionCosts, annotations=slice(None)):
        self.annotations.loc[annotations, "connection_cost"] = connectionCosts

    def resetPruningLabels(self):
        self.annotations.is_pruned = False
        self.annotations.is_merged = False
        self.annotations.prune_distance = np.nan
