import numpy as np
import pandas as pd
import scipy as sp

import numba
import logging
import itertools
import pickle
import os
import collections

import logging
import sys

from FacilityLocation import FacilityLocation, FData, CData
from AnnotationStore import AnnotationStore
from StatisticStore import StatisticStore
from ImageProcessor import ImageProcessor
from ImageView import ImageView
from WorkerView import WorkerView


@numba.jit(nopython=True, cache=True)
def computeIouDistancesAsymm(normedGtBoxCoords, normedBoxCoords):

    distances = np.full(
        (normedGtBoxCoords.shape[0], normedBoxCoords.shape[0]), np.nan, dtype=np.float32
    )

    for left in range(distances.shape[0]):
        x1l, x2l, y1l, y2l = normedGtBoxCoords[left]
        # distances[left, left] = np.nan
        for right in range(distances.shape[1]):
            x1r, x2r, y1r, y2r = normedBoxCoords[right]
            # Maximised @ 1 when IoU = 0, i.e. no overlap
            distances[left, right] = 1.0 - (
                (
                    max(0, min(x2l, x2r) - max(x1l, x1r))
                    * max(0, min(y2l, y2r) - max(y1l, y1r))
                )
                / (
                    max(
                        1e-8,
                        (max(x2l, x2r) - min(x1l, x1r))
                        * (max(y2l, y2r) - min(y1l, y1r)),
                    )
                )
            )
    return distances


@numba.jit(nopython=True, cache=True)
def computeDisallowedConnectionMask(
    exclusiveIndices, exclusiveIndexSetSizes, maskShape
):
    # Let the zeroth row and column pertain to the dummy facility
    mask = np.ones((maskShape[0] + 1, maskShape[1] + 1))

    begin = 0
    for exclusiveIndexSetSize in exclusiveIndexSetSizes:
        end = begin + exclusiveIndexSetSize
        for left in exclusiveIndices[begin:end]:
            for right in exclusiveIndices[begin:end]:
                # Value of zero implies that the connection is diallowed
                mask[left + 1, right + 1] = 0
        begin = end

    return mask


@numba.jit(nopython=True, cache=True)
def computeInitConnectionCostsImpl(distances, threshold=0.5):
    connectionCosts = np.empty(
        (distances.shape[0] + 1, distances.shape[1] + 1), dtype=np.float32
    )
    # Let the zeroth column represent the dummy facility
    connectionCosts[0, 0] = 0
    connectionCosts[0, 1:] = 1.0
    connectionCosts[1:, 0] = 1.0

    for left in range(distances.shape[0]):
        for right in range(distances.shape[1]):
            if left == right:
                connectionCosts[left + 1, right + 1] = 0
            else:
                connectionCosts[left + 1, right + 1] = (
                    0 if distances[left, right] < threshold else np.nan
                )

    return connectionCosts


# Computes the cost of connecting a box ("city", left) to a ground truth box ("facility", right)
# Sigmas, and probabilities are only supplied for the "city".
@numba.jit(nopython=True, cache=True)
def computeConnectionCostsImpl(distances, falsePosProbs, falseNegProbs, variances):
    connectionCosts = np.empty(
        (distances.shape[0] + 1, distances.shape[1] + 1), dtype=np.float32
    )
    # Let the zeroth row represent the dummy facility
    connectionCosts[0, 0] = 0
    connectionCosts[0, 1:] = -np.log(falsePosProbs)
    connectionCosts[1:, 0] = connectionCosts[0, 1:]

    halfLogTwoPi = 0.5 * np.log(2 * np.pi)
    for left in range(distances.shape[0]):
        # Factor of 0.5 converts variance to std-dev
        connectionCost = (
            0.5 * np.log(variances[left])
            + np.log(
                falseNegProbs[left]
                / ((1 - falsePosProbs[left]) * (1 - falseNegProbs[left]))
            )
            + halfLogTwoPi
        )
        for right in range(distances.shape[1]):
            connectionCosts[left + 1, right + 1] = (
                (0.5 * (distances[left, right] ** 4) / variances[left] + connectionCost)
                if distances[left, right] < 1
                else np.nan
            )
    return connectionCosts


@numba.jit(nopython=True, cache=True)
def mergeAssociationsImpl(group):
    merged = (group[:, :-3].T / group[:, -3]).sum(axis=1) / (
        np.reciprocal(group[:, -3]).sum()
    )
    normedMerged = np.concatenate(
        (merged[:2] / group[0, -2], merged[2:] / group[0, -1])
    )
    out = np.concatenate((merged, normedMerged))
    return out


# Compute the IoU distance between all associated boxes and the mean ground truths
@numba.jit(nopython=True, cache=True)
def computeIouWithGroundTruth(
    annotationBoxCoords, groundTruthBoxCoords, groundTruthIndexSetSizes
):

    distances = np.empty((annotationBoxCoords.shape[0], 1), dtype=np.float32)

    begin = 0
    for groundTruth in range(groundTruthBoxCoords.shape[0]):
        end = begin + groundTruthIndexSetSizes[groundTruth]
        x1l, x2l, y1l, y2l = groundTruthBoxCoords[groundTruth]
        for annotation in range(begin, end):
            x1r, x2r, y1r, y2r = annotationBoxCoords[annotation]
            # Maximised @ 1 when IoU = 0, i.e. no overlap
            distances[annotation] = 1.0 - (
                (
                    max(0, min(x2l, x2r) - max(x1l, x1r))
                    * max(0, min(y2l, y2r) - max(y1l, y1r))
                )
                / (
                    max(
                        1e-8,
                        (max(x2l, x2r) - min(x1l, x1r))
                        * (max(y2l, y2r) - min(y1l, y1r)),
                    )
                )
            )
            begin = end
    return distances


class BoxAggregator:

    defaultDatasetWidePriorInitialValues = dict(
        volunteer_skill=dict(false_pos_prob=0.01, false_neg_prob=0.1, variance=0.5),
        image_difficulty=dict(variance=0.5),
        # shared=dict(variance=0.5),
    )

    defaultDatasetWidePriorParameters = dict(
        volunteer_skill=dict(
            nBeta_false_pos=500, nBeta_false_neg=10, nInv_chisq_variance=10
        ),
        image_difficulty=dict(nInv_chisq_variance=10),
        shared=dict(max_num_ground_truths=20),
    )

    # Note required_overlap_fraction is a bit of a misnomer. In fact it is
    # interpreted as the maximum allowable 1-IoU distance, so it is really more
    # like the reciprocal of the required overlap fraction.
    defaultInitPhaseParams = dict(
        required_multiplicity_fraction=0.1, required_overlap_fraction=0.6
    )

    defaultFilterParams = dict(min_images_per_worker=10, min_workers_per_image=5)

    defaultImageCompletionParams = dict(
        max_risk=1,
        max_expected_num_false_pos=0.3,
        max_expected_num_false_neg=0.3,
        max_expected_num_inaccurate=0.3,
        max_num_batches_not_finished=10,
    )

    defaultLossParams = dict(
        false_pos_loss=1,
        false_neg_loss=1,
        required_accuracy=0.2,
        single_box_filter_threshold=None,
    )

    defaultAssocParams = dict(
        prune_ground_truths=False,
        prune_attempt_merge=False,
        merge_ground_truths=False,
        merge_ground_truth_threshold=0.3,
    )

    boxCoordCols = ["x1", "x2", "y1", "y2"]
    normedBoxCoordCols = ["x1_normed", "x2_normed", "y1_normed", "y2_normed"]

    def __init__(
        self,
        imageCompletionAssessor=None,
        imageCompletionParams=None,
        applyInputDataFilter=False,
        inputDataFilter=None,
        filterParams=None,
        initPhaseParams=None,
        lossParams=None,
        assocParams=None,
        datasetWidePriorInitialValues=None,
        datasetWidePriorParameters=None,
        savePath=None,
        computeStepwiseRisks=False,
        maxBatchIterations=10,
        logFile=None,
    ):
        self.setupLogging(logFile)

        self.datasetWidePriorInitialValues = (
            BoxAggregator.defaultDatasetWidePriorInitialValues
        )
        if datasetWidePriorInitialValues is not None:
            self.datasetWidePriorInitialValues.update(datasetWidePriorInitialValues)

        self.datasetWidePriorParameters = (
            BoxAggregator.defaultDatasetWidePriorParameters
        )
        if datasetWidePriorInitialValues is not None:
            self.datasetWidePriorParameters.update(datasetWidePriorParameters)

        self.initPhaseParameters = BoxAggregator.defaultInitPhaseParams
        if initPhaseParams is not None:
            self.initPhaseParameters.update(initPhaseParams)

        self.applyInputDataFilter = applyInputDataFilter
        self.inputDataFilter = inputDataFilter

        self.filterParams = BoxAggregator.defaultFilterParams
        if filterParams is not None:
            self.filterParams.update(filterParams)

        self.imageCompletionAssessor = (
            imageCompletionAssessor
            if imageCompletionAssessor is not None
            else self.assessImageCompletion
        )

        self.imageCompletionParams = BoxAggregator.defaultImageCompletionParams
        if imageCompletionParams is not None:
            self.imageCompletionParams.update(imageCompletionParams)

        self.lossParams = BoxAggregator.defaultLossParams
        if lossParams is not None:
            self.lossParams.update(lossParams)

        self.assocParams = BoxAggregator.defaultAssocParams
        if assocParams is not None:
            self.assocParams.update(assocParams)

        self.savePath = savePath
        # Computationally expensive, but can be useful for diagnostics
        self.computeStepwiseRisks = computeStepwiseRisks
        self.maxBatchIterations = maxBatchIterations

        # Class that implements image-specific computations
        self.imageProcessor = ImageProcessor()

        # Save settings for this run.
        if self.savePath is not None:
            self.saveSettings(os.path.join(self.savePath, f"settings.pkl"))

        self.setup()

    def filterInputData(self, dataBatch, filterParams):
        return (
            dataBatch.groupby(by=["worker_id"])
            .filter(
                lambda x: (
                    x.image_id.unique().size > filterParams["min_images_per_worker"]
                )
                and workerFilter(x.name)
            )
            .groupby(by=["image_id"])
            .filter(
                lambda x: x.worker_id.unique().size
                > filterParams["min_workers_per_image"]
            )
        ).reset_index(drop=True)

    def assessImageCompletion(
        self, imageStats: pd.DataFrame, params: dict
    ) -> pd.Series:
        """Simple default image completion assessor checks whether the image
        statistics corresponding to the keys of params are less than the values
        of params.

        Parameters
        ----------
        imageStats : pd.DataFrame
            The computed image statistics.
        params : dict
            Threshold values for specific named image statistics.

        Returns
        -------
        pd.Series
            True if all statistics for an image are less than their specified
            threshold values. False otherwise.

        """
        orderedParams = collections.OrderedDict(params)
        columns = list(map(lambda key: key.replace("max_", ""), orderedParams.keys()))
        result = np.all(
            imageStats.loc[:, columns]
            < np.fromiter(orderedParams.values(), dtype=float),
            axis=1,
        )
        return result

    def setup(self):
        self.annoStore = AnnotationStore(logFunction=self.printToLog)
        self.statStore = StatisticStore(logFunction=self.printToLog)
        self.batchLikelihoods = []
        self.batchCounter = 0
        self.finishedImageRunningTotal = 0

    def setupLogging(self, logFile=None):
        self.logger = logging.getLogger("BoxAggregatorLogger")
        self.logger.setLevel(logging.INFO)

        if logFile is not None:
            self.logger.handlers = []
            self.logger.propagate = False
            logHandler = logging.FileHandler(logFile)
            logHandler.setLevel(logging.INFO)
            self.logger.addHandler(logHandler)

    def printToLog(self, *args, logtype="info", sep=" "):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def setupNewBatch(self, dataBatch):
        self.batchCounter += 1
        self.dataBatch = (
            self.inputDataFilter(dataBatch, self.filterParams)
            if self.inputDataFilter is not None
            else self.filterInputData(dataBatch, self.filterParams)
            if self.applyInputDataFilter
            else dataBatch
        )

        # Add new annotations and initialise with dataset-wide prior values
        self.annoStore.addAnnotations(
            self.dataBatch,
            self.datasetWidePriorInitialValues,
            statisticStore=self.statStore,
        )
        self.images = self.annoStore.generateViews(
            "image_id", ImageView, self.statStore
        )
        self.workers = self.annoStore.generateViews(
            "worker_id", WorkerView, self.statStore
        )

        self.statStore.addAnnotations(
            self.annoStore,
            self.datasetWidePriorInitialValues,
            self.datasetWidePriorParameters,
        )

        self.batchLikelihoods.append([])
        self.likelihoods = self.batchLikelihoods[-1]

    def computeConnectionCosts(
        self,
        distances,
        falsePosProbs=None,
        falseNegProbs=None,
        variances=None,
        init=False,
        initThreshold=0.95,
    ):
        if init:
            return computeInitConnectionCostsImpl(distances, initThreshold)
        elif (
            falsePosProbs is not None
            and falseNegProbs is not None
            and variances is not None
        ):
            return computeConnectionCostsImpl(
                distances, falsePosProbs, falseNegProbs, variances
            )

    def computeAssociations(
        self, image, openCosts, connectionCosts, disallowedConnectionMask, verbose=False
    ):
        # indices are needed to index np arrays
        annotationIndices = image.getAnnotationIndices()
        # labels are needed to index pandas dataframes using .loc[]
        annotationIndexLabels = image.getAnnotationIndexLabels()
        # Filter empty annotations
        validAnnoSelector = ~image.getAnnotations()["empty"]
        annotationIndices = annotationIndices[validAnnoSelector]
        annotationIndexLabels = annotationIndexLabels[validAnnoSelector]

        # Select the dummy (0) and all annotations associated with this image
        # Note the +1 offset accounts for the dummy in the zeroth row/column.
        selectedAnnotationIndices = [0] + (annotationIndices + 1).tolist()
        selectedAnnotationIndexLabels = [-1] + (annotationIndexLabels).tolist()

        # ix_ builds an indexer array to extract a 2D subset of connection costs
        selector = np.ix_(selectedAnnotationIndices, selectedAnnotationIndices)
        fl = FacilityLocation(
            np.nan_to_num([0] + openCosts.to_numpy()[annotationIndices, 1].tolist()),
            connectionCosts[selector],
            disallowedConnectionMask.to_numpy()[selector].astype(bool),
            logging.WARNING,
        )
        fl.solve(haveDummy=True)

        image.annotationStore.annotations.loc[
            annotationIndexLabels, "is_ground_truth"
        ] = fl.fData.data[1:, FData.isOpen].astype(bool, copy=False)

        image.annotationStore.annotations.loc[
            annotationIndexLabels, "is_cleanup_ground_truth"
        ] = fl.fData.data[1:, FData.isCleanup].astype(bool, copy=False)

        image.annotationStore.annotations.loc[annotationIndexLabels, "association"] = (
            fl.cData.data[1:, CData.facility] - 1
        )

        # To select the connection costs for cities (to a facility or the dummy)
        # index using the open facilities connected to and all cities
        connectedCosts = connectionCosts[selector][  # includes dummy
            fl.cData.data[1:, CData.facility].astype(
                int, copy=False
            ),  # selects rows for facilities
            range(1, fl.cData.data.shape[0]),  # selects all but dummy city
        ]

        # if verbose:
        #     if image.imageId == 36718317:
        #         self.printToLog(
        #             "C => P > 1",
        #             image.imageId,
        #             annotationIndexLabels,
        #             selector,
        #             # np.exp(-connectedCosts),
        #             sep="\n",
        #         )
        #         self.printToLog(
        #             connectionCosts[selector],
        #             *zip(
        #                 fl.cData.data[1:, CData.facility].astype(int),
        #                 list(range(1, fl.cData.data.shape[0])),
        #                 connectedCosts,
        #             ),
        #             # np.exp(-connectedCosts),
        #             sep="\n",
        #         )
        #
        #         self.printToLog(
        #             "Before",
        #             *zip(
        #                 fl.cData.data[1:, CData.facility].astype(int),
        #                 list(range(1, fl.cData.data.shape[0])),
        #                 annotationIndexLabels,
        #                 image.annotationStore.annotations.loc[
        #                     annotationIndexLabels, ["connection_cost", "association"]
        #                 ].to_numpy(),
        #             ),
        #             sep="\n",
        #         )
        image.annotationStore.annotations.loc[
            annotationIndexLabels, "connection_cost"
        ] = connectedCosts
        # self.printToLog(
        #     "After",
        #     *zip(
        #         fl.cData.data[1:, CData.facility].astype(int),
        #         list(range(1, fl.cData.data.shape[0])),
        #         annotationIndexLabels,
        #         image.annotationStore.annotations.loc[
        #             annotationIndexLabels, ["connection_cost", "association"]
        #         ].to_numpy(),
        #     ),
        #     sep="\n",
        # )
        # Ensure that ground-truth boxes are associated with themselves
        # The facility location algorithm leaves opened facilities with the
        # dummy association index.
        gtAnnotationIndices = np.flatnonzero(
            image.annotationStore.annotations.loc[
                annotationIndexLabels, "is_ground_truth"
            ]
        )
        image.annotationStore.annotations.loc[
            annotationIndexLabels[gtAnnotationIndices], "association"
        ] = gtAnnotationIndices

        # Require that the associated facility is open and is not the dummy.
        image.annotationStore.annotations.loc[
            annotationIndexLabels, "matches_ground_truth"
        ] = (
            image.annotationStore.annotations.loc[annotationIndexLabels, "association"]
            >= 0
        )

    def mergeAssociations(self):
        """Compute average bounding boxes from all boxes associated with a
        particular ground truth.

        Returns
        -------
        NoneType

        """
        # Can work on a per image basis. Group annotations into image-specific
        # association groups.
        # Note that the dummy facility is ignored.
        associatedBoxes = self.annoStore.annotations.loc[
            self.annoStore.annotations.matches_ground_truth
        ]

        boxGrouper = associatedBoxes.groupby(by=["image_id", "association"])

        # For each group, compute a "mean box". The accuracy of each associated annotation
        # will be evaluated by computing 1-IoU with this "mean box".
        self.groundTruthBoxCoordsFrame = boxGrouper[
            BoxAggregator.boxCoordCols + ["variance", "image_width", "image_height"]
        ].apply(lambda group: pd.Series(mergeAssociationsImpl(group.to_numpy())))
        self.groundTruthBoxCoordsFrame.columns = (
            self.boxCoordCols + self.normedBoxCoordCols
        )
        self.groundTruthBoxCoords = self.groundTruthBoxCoordsFrame.to_numpy()
        # Store image IDs associated with GT boxes. Needed for false negative
        # count prediction.
        self.groundTruthBoxImageIds = boxGrouper.first().index.get_level_values(0)

        # Compute the size of each group associated with a "mean box".
        # This will be used to determine which annotations should be compared
        # with each "mean box" for JIT-compiled distance computation.
        # Note: choice of group worker_id column is somewhat arbitrary, but
        # ensures no NaNs
        self.groundTruthIndexSetSizes = boxGrouper.count().worker_id.to_numpy()

        # Extract the box coordinates for all annotations not associated
        # with the dummy facility and sort them so that their order corresponds
        # with that of the mean boxes and set sizes i.e. sort to match the
        # grouper sorting.
        self.annotationBoxCoords = associatedBoxes.sort_values(
            by=["image_id", "association"]
        )[BoxAggregator.boxCoordCols]

    def pruneGroundTruths(
        self, attemptMerge=False, mergeThreshold=0.5, isolatedFalsePosProbThreshold=0.9
    ):
        """The facility location algorithm prevents facilities from closing once
        they are open. When a large number of annotations are available, this
        can lead to single boxes being defined as facilities then subsequently
        having all their contributing cities switch, leaving them isolated.

        These isolated boxes typically have false_pos_prob very close to 1.

        This method implements two possible approaches to mitigating this issue.

        1) Simply assign the boxes to the dummy facility and remove them from
        the ground truth set.
        2) If possible, merge the box with another ground truth that it overlaps
        with.

        Parameters
        ----------
        attemptMerge : bool
            If true, search for ground truth boxesthat overlap and assign
            isolated boxes to that association.
            If False, assign the isolated box to the dummy and remove it from
            the ground truth set.
        mergeThreshold : float
            The maximum 1-IoU distance between an isolated box and a corresponding
            associated box for merging to be allowed.
        isolatedFalsePosProbThreshold : float
            The minimum false positive probability for an isolated ground truth
            box to be retained, despite having only one contributing annotation.

        Returns
        -------
        NoneType

        """
        self.printToLog("\tPruning/merging isolated boxes...")
        self.annoStore.resetPruningLabels()
        # Select valid annotations that are not associated with the dummy
        validAnnos = self.annoStore.annotations[
            (self.annoStore.annotations.matches_ground_truth)
            & ~self.annoStore.annotations["empty"]
        ]
        # Find isolated ground boxes. Note that we select based on
        # matches_ground_truth, since this includes is_ground_truth.
        # We expect most isolated boxes to be defined as ground truths.
        isolated = (
            validAnnos.groupby(["image_id", "association"])
            .filter(lambda grp: grp.index.size <= 1)
            .index
        )
        self.printToLog(f"\t\tFound {isolated.size} isolated boxes")

        if isolated.size > 0 and attemptMerge:
            self.printToLog(
                "\t\tAttempting to merge isolated boxes to ground truths..."
            )
            # Find ground truth boxes to match against
            associated = validAnnos.groupby(["image_id", "association"]).filter(
                lambda grp: grp.index.size > 1
            )
            associated = associated[associated.is_ground_truth].index
            # Determine whether any isolated boxes are in an image that has
            # at least one ground truth box.
            matchable = self.annoStore.annotations.loc[isolated].image_id.isin(
                self.annoStore.annotations.loc[associated].image_id
            )

            # If any match candidates were provided by the same worker as the
            # isolated box, remove them.

            # Get the distance between the isolated boxes and their potential
            # associations.
            distances = self.distances.loc[
                matchable[matchable].index.to_list(), associated.to_list()
            ]

            # Retrieve the appropriate segment of the disallowed connection mask
            # Connection mask value of zero implies disallowed connection
            mask = (
                self.disallowedConnectionMask.loc[
                    associated.to_list(), matchable[matchable].index.to_list()
                ]
                < 1
            )
            # Prevent matching any boxes provided by the same worker for the same
            # image by setting the corresponding distance to infinity.
            self.printToLog(
                f"\t\t\tDisallowing {mask.sum().sum()} merge candidates provided by the same worker for the same image"
            )
            # Mask replaces anywhere disallowed >= 1
            distances = distances.mask(mask, np.infty)

            # Compute the index of an associated box that minimizes the distance to
            # an isolated ground truth within each image
            rawClosest = distances.groupby(
                by=self.annoStore.annotations.loc[
                    matchable[matchable].index.to_list()
                ].image_id
            ).apply(
                lambda grp: grp.T.groupby(
                    self.annoStore.annotations.loc[associated.to_list()].image_id
                )
                .get_group(grp.name)
                .idxmin(axis=0)
            )
            try:
                closest = rawClosest.loc[np.isfinite(rawClosest)]
            except ValueError as e:
                print(e, rawClosest, sep="\n")
            if closest.shape[0] > 0:
                matchCandidates = (
                    pd.concat(
                        [
                            closest,
                            pd.Series(
                                self.distances.lookup(
                                    *(closest.reset_index(level=1).to_numpy().T)
                                ),
                                index=closest.index,
                            ),
                        ],
                        axis=1,
                    )
                    .reset_index(level=1)
                    .rename(
                        columns={0: "closest", 1: "distance", "level_1": "isolated"}
                    )
                )

                # Identify isolated boxes that are close enough to associations to merge
                matchCandidates["overlaps"] = matchCandidates.distance < mergeThreshold
                if matchCandidates.overlaps.any():
                    self.printToLog(
                        f"\t\t{matchCandidates.overlaps.sum()} Viable merge candidates found."
                    )
                else:
                    self.printToLog(f"\t\tNo viable merge candidates found.")

                # Correct associations appropriately
                self.annoStore.annotations.loc[
                    matchCandidates.isolated.to_list(), "is_pruned"
                ] = True
                self.annoStore.annotations.loc[
                    matchCandidates.isolated.to_list(), "prune_distance"
                ] = matchCandidates.distance.to_numpy()
                self.annoStore.annotations.loc[
                    matchCandidates[matchCandidates.overlaps].isolated.to_list(),
                    "is_merged",
                ] = True
                self.annoStore.annotations.loc[
                    matchCandidates[matchCandidates.overlaps].isolated.to_list(),
                    "association",
                ] = self.annoStore.annotations.loc[
                    matchCandidates[matchCandidates.overlaps].closest.to_list(),
                    "association",
                ].to_numpy()
                self.annoStore.annotations.loc[
                    matchCandidates[matchCandidates.overlaps].isolated.to_list(),
                    "is_ground_truth",
                ] = False
                self.annoStore.annotations.loc[
                    matchCandidates[matchCandidates.overlaps].isolated.to_list(),
                    "matches_ground_truth",
                ] = True

                self.annoStore.annotations.loc[
                    matchCandidates[~matchCandidates.overlaps].isolated.to_list(),
                    "association",
                ] = -1
                self.annoStore.annotations.loc[
                    matchCandidates[~matchCandidates.overlaps].isolated.to_list(),
                    "is_ground_truth",
                ] = False
                self.annoStore.annotations.loc[
                    matchCandidates[~matchCandidates.overlaps].isolated.to_list(),
                    "matches_ground_truth",
                ] = False
                return
            else:
                self.printToLog(f"\t\tNo viable merge candidates found.")
        if isolated.size > 0:
            self.printToLog("\t\tAssigning isolated boxes to the dummy...")
            self.annoStore.annotations.loc[isolated, "is_pruned"] = True
            self.annoStore.annotations.loc[isolated, "association"] = -1
            self.annoStore.annotations.loc[isolated, "is_ground_truth"] = False
            self.annoStore.annotations.loc[isolated, "matches_ground_truth"] = False

    def mergeGroundTruths(self, mergeThreshold=0.5):
        """After pruning it may still be the case that two ground truth boxes
        overlap by a significant fraction. Make a pass over the association data
        and merge these boxes and all their associated boxes together.

        Parameters
        ----------
        mergeThreshold : float
            The maximum 1-IoU distance between two ground truths that may be
            combined.

        Returns
        -------
        NoneType

        """
        self.printToLog("\tMerging overlapping ground truth boxes...")
        # Find indices of gt boxes from images with more than one ground truth
        mergeableGtBoxes = (
            self.annoStore.annotations[self.annoStore.annotations.is_ground_truth]
            .groupby(by="image_id")
            .filter(lambda grp: grp.index.size > 1)
        )
        # Extract the precomputed distances for applicable indices
        gtBoxDistances = self.distances.loc[
            mergeableGtBoxes.index, mergeableGtBoxes.index
        ]

        # As in the prune stage, mask any disallowed connections
        # Retrieve the appropriate segment of the disallowed connection mask
        # Connection mask value of zero implies disallowed connection
        mask = (
            self.disallowedConnectionMask.loc[
                mergeableGtBoxes.index, mergeableGtBoxes.index
            ]
            < 0.5
        )
        # Prevent matching any boxes provided by the same worker for the same
        # image by setting the corresponding distance to infinity.
        self.printToLog(
            f"\t\tDisallowing {mask.sum().sum()} ground truth merge "
            "candidates provided by the same worker for the same image"
        )
        # Mask replaces anywhere disallowed >= 1
        gtBoxDistances = gtBoxDistances.mask(mask, np.infty)

        # For each image find any pairs of gtBoxes that are closer than mergeThreshold
        # Find lists of potential merge candidates for each GT within each specific image
        gtBoxDistances.index.name = "box_id"
        mergeLists = (
            gtBoxDistances.groupby(by=mergeableGtBoxes.image_id)
            .apply(
                # Outer group has all box ids as columns and box labels for specific
                # image as rows. Transpose it and group the boxes again
                lambda grp: grp.T.groupby(by=mergeableGtBoxes.image_id)
                .get_group(grp.name)  # get the group that matches the *outer* image_id
                .lt(mergeThreshold)  # mark boxes that are close enough to merge
                # extract the ids of boxes that are close enough to merge
                .apply(lambda s: s.index[np.flatnonzero(s)].to_numpy(), axis=1)
            )
            .rename("merge_list")
        )

        if not mergeLists.empty:
            # remove bidirectional merges. The order is not important because
            # the final GT box is the mean of *all* boxes in the association.
            try:
                filteredMergeLists = pd.Series(
                    # bring the box labels into the space of addressable data as "box_id"
                    mergeLists.reset_index()
                    .apply(
                        # The column name
                        lambda mergeList: mergeList.merge_list[
                            mergeList.merge_list < mergeList.box_id
                        ],
                        axis=1,
                    )
                    .to_numpy(),
                    index=mergeLists.index,
                ).rename("filtered_merge_list")
            except Exception as e:
                self.printToLog(
                    "Exception filtering ground truth merge lists...",
                    e,
                    mergeLists.reset_index()
                    .apply(
                        # The column name
                        lambda mergeList: mergeList.merge_list[
                            mergeList.merge_list < mergeList.box_id
                        ],
                        axis=1,
                    )
                    .to_numpy(),
                    mergeLists,
                    logtype="warning",
                    sep="\n",
                )

            # Collect all information required for merge
            finalMergeLists = pd.concat(
                [
                    filteredMergeLists,
                    (filteredMergeLists.apply(len) <= 0).rename("merge_list_empty"),
                    filteredMergeLists.apply(
                        lambda mergees: self.annoStore.annotations.loc[
                            mergees, "association"
                        ].to_numpy()
                    ).rename("mergee_association"),
                    pd.Series(
                        self.annoStore.annotations.loc[
                            filteredMergeLists.index.get_level_values(1), "association"
                        ].to_numpy(),
                        index=filteredMergeLists.index,
                    ).rename("target_association"),
                ],
                axis=1,
            )

            # If there are no merge candidates, stop here
            if np.all(finalMergeLists.merge_list_empty):
                self.printToLog(
                    "\t\tmergeGroundTruths: After filtering, no"
                    + " merge candidates remain"
                )
                return
            elif "mergee_association" not in finalMergeLists.columns:
                self.printToLog("\t\tmergeGroundTruths: No mergee_association")
                return
            elif "target_association" not in finalMergeLists.columns:
                self.printToLog("\t\tmergeGroundTruths: No target_association")
                return

            # Construct a dataframe with an index enumerating the boxes to be
            # merged and a column containing the box they should be merged into
            mergeTargetFrame = pd.concat(
                finalMergeLists[~finalMergeLists.merge_list_empty]
                .reset_index()
                .apply(
                    lambda row: pd.DataFrame(
                        {
                            "target": row.box_id,
                            "target_association": row.target_association,
                        },
                        index=[
                            pd.Series(row.filtered_merge_list, name="mergee"),
                            pd.Series(
                                [row.image_id] * row.filtered_merge_list.size,
                                name="image_id",
                            ),
                            pd.Series(
                                row.mergee_association, name="mergee_association"
                            ),
                        ],
                    ),
                    axis=1,
                )
                .to_numpy()
            )

            # Now find all annotations that belong to the mergee
            selector = (
                self.annoStore.annotations.loc[:, ["image_id", "association"]]
                .apply(tuple, axis=1)
                .isin(mergeTargetFrame.index.droplevel(0))
                .to_numpy()
            )

            # TODO: All this checking can probably be removed now.
            def printingExtract(row, mtf):
                try:
                    if mtf.index.size == 0:
                        self.printToLog("Zero-length index", mtf.columns, res, sep="\n")
                    res = mtf.loc[(row.image_id, row.association)]
                    if res.ndim > 1:
                        res = res.head(1).squeeze()
                    if len(res.shape) != 1:
                        self.printToLog(
                            "Not 1-dimensional",
                            res.shape,
                            res,
                            mtf.loc[(row.image_id, row.association)],
                            mtf.loc[(row.image_id, row.association)].head(1),
                            sep="\n\n",
                        )

                    return res  # mtf.loc[(row.image_id, row.association)]
                except KeyError as e:
                    self.printToLog(
                        e, row.image_id, row.association, mtf, mtf.index, sep="\n"
                    )
                    raise e
                except AttributeError as e:
                    self.printToLog(
                        e, row.image_id, row.association, mtf, mtf.index, sep="\n"
                    )
                    raise e

            if not np.any(selector):
                self.printToLog("\t\tNo matching annotations!")
            else:
                self.printToLog(
                    f"\t\tmergeGroundTruths: {selector.sum()} merge "
                    + "candidates remaining"
                )

                mtf = mergeTargetFrame.reset_index(level=0).sort_index()
                targetAssociations = self.annoStore.annotations.loc[
                    selector, ["image_id", "association"]
                ].apply(
                    printingExtract,  # lambda row, mtf: mtf.loc[(row.image_id, row.association)],
                    axis=1,
                    args=(mtf,),
                )
                try:
                    exampleTargets = self.annoStore.annotations.loc[
                        selector, "association"
                    ]
                    self.printToLog(
                        "\t\tmergeGroundTruths: Merging "
                        + f"{len(exampleTargets)} ground truth boxes to "
                        + f"{len(exampleTargets.unique())} targets"
                    )
                    # self.printToLog(f"{exampleTargets} -> ")
                    # self.printToLog(
                    #     f"{targetAssociations.target_association.to_numpy()}"
                    # )
                    self.annoStore.annotations.loc[
                        selector, "association"
                    ] = targetAssociations.target_association.to_numpy()
                    # exampleTargets = self.annoStore.annotations.loc[
                    #     selector, "association"
                    # ]
                    # self.printToLog("\t\tmergeGroundTruths: Post-merge")
                    # self.printToLog(f"{exampleTargets}")
                except AttributeError as e:
                    self.printToLog(
                        e,
                        selector,
                        np.flatnonzero(selector),
                        mergeTargetFrame,
                        targetAssociations,
                        targetAssociations.iloc[0],
                        self.annoStore.annotations.loc[
                            selector, ["image_id", "association"]
                        ],
                        sep="\n\n",
                    )

    def computeWorkerSkills(self, worker, onlyFinished=False, verbose=False):
        # worker annotations - only keep new annotations as old ones are already
        # counted in the prior.
        annos = worker.getAnnotations()
        annotationIndexLabels = worker.getAnnotationIndexLabels()

        selector = None
        if onlyFinished:
            finishedImages = self.statStore.imageStatistics.index[
                self.statStore.imageStatistics.is_finished
            ].intersection(annos.image_id.unique())
            selector = annos.image_id.isin(finishedImages)
            # Selecting finished annos also guarantees that only finished
            # images will be included later in workerImageAnnos.
            annos = annos.loc[selector]
            annotationIndexLabels = annotationIndexLabels[selector]

        # ** Raw annotation counts
        numAnnos = annos.index.size

        if numAnnos > 0:
            # worker annotations grouped by image
            imageGroupedAnnos = annos.groupby(by="image_id")
            # Images seen by this worker
            workerImages = annos.image_id.unique()

            # *All* annotations for images seen by this worker i.e. even if the
            # worker left it blank.
            workerImageAnnos = worker.annotationStore.annotations.loc[
                pd.Index(self.annoStore.annotations["image_id"]).isin(workerImages)
            ]
            # Number of workers annotating each image
            numWorkersPerImage = workerImageAnnos.groupby(
                by="image_id"
            ).worker_id.nunique()

            # Workers are assigned a weight to account for the fact that they may
            # be one of many or few that annotated the image. If their assessment
            # agrees/disagrees with a large number of annotators then the
            # increment/decrement to their skill is enhanced.
            multiplicityWeights = (
                (numWorkersPerImage - 1) / numWorkersPerImage
            ).rename("weights")
            multiplicityWeights = multiplicityWeights.where(
                multiplicityWeights > 0, np.finfo(np.float64).tiny
            )

            workerImageAnnos = workerImageAnnos.merge(
                multiplicityWeights, how="left", left_on="image_id", right_index=True
            )  # Now a second copy detatched from annoStore

            worker.annotationStore.annotations.loc[
                annotationIndexLabels, "multiplicity_weights"
            ] = workerImageAnnos.loc[annos.index.to_numpy(), "weights"]

            # Reload updated annos
            annos = worker.getAnnotations()
            if onlyFinished:
                annos = annos.loc[selector]

            # Compute annotation statistics

            # ** Weighted annotation counts
            numWeightedAnnos = annos.multiplicity_weights.sum()

            # ** False positives
            numFalsePosAnnos = (
                annos.multiplicity_weights * ~annos.matches_ground_truth
            ).sum()

            # ** False negatives

            # Summed number of ground truth boxes in images seen by this worker
            numGroundTruths = (
                workerImageAnnos.is_ground_truth * workerImageAnnos.weights
            ).sum()
            # Summed number of annotations by this worker that match a ground truth
            numMatchesGroundTruths = (
                annos.multiplicity_weights
                * annos.matches_ground_truth
                * ~annos.is_merged
            ).sum()
            # The number they missed i.e. number available - number matched
            numMissedGroundTruths = numGroundTruths - numMatchesGroundTruths
            if numMissedGroundTruths < -1e-8:
                self.printToLog(
                    f"Worker {worker.workerId}.\n"
                    "Apparently negative number of missed ground truths!\n",
                    f"{numGroundTruths}, {numMatchesGroundTruths}, {numMissedGroundTruths}",
                    logtype="warning",
                )
                if self.savePath is not None:
                    filePath = os.path.join(
                        self.savePath, f"worker_{worker.workerId}_nmgt.pkl"
                    )
                    with open(filePath, mode="wb") as nmgtFile:
                        self.printToLog(
                            f"Saving worker annotation data for worker {worker.workerId} to {filePath}.",
                            logtype="warning",
                        )
                        pickle.dump(
                            dict(
                                annos=worker.getAnnotations(),
                                image_annos=workerImageAnnos,
                                finished_selector=selector,
                            ),
                            nmgtFile,
                        )

            # ** Variances
            deltaVariance = (
                np.nansum(annos.ground_truth_distance ** 2)
                + (
                    self.statStore.imageStatistics.loc[
                        imageGroupedAnnos.groups.keys(), "variance"
                    ]  # This is why image params must be computed first!
                    * imageGroupedAnnos.matches_ground_truth.sum()
                ).sum()
            )

            # ** Save computed count and variance statistics for this worker
            worker.incrementStatistics(
                # the actual number of annotations
                numAnnos,
                # the effective number of trials that could yield false positive
                numWeightedAnnos,
                # the actual number of false positives
                numFalsePosAnnos,
                # the effective number of trials that could yield false negative
                numGroundTruths,
                # the actual number of false negatives
                numMissedGroundTruths,
                # the number of matched ground truths
                numMatchesGroundTruths,
                # the variance update appropriate for the matched ground truths
                deltaVariance,
            )

            # Log when a volunteer annotates more than 10 clumps in an image
            # This might indicate a mistake.
            if imageGroupedAnnos.size().max() > 10:
                self.printToLog(
                    f"Worker {worker.workerId}:",
                    "Maximum marks/image > 10",
                    imageGroupedAnnos.ngroups,
                    imageGroupedAnnos.size().describe(),
                    numAnnos,
                    # the effective number of trials that could yield false positive
                    numWeightedAnnos,
                    # the actual number of false positives
                    numFalsePosAnnos,  # * regularisingFactor,
                    # the effective number of trials that could yield false negative
                    numGroundTruths,
                    # the actual number of false negatives
                    numMissedGroundTruths,
                    # the number of matched ground truths
                    numMatchesGroundTruths,
                    # the variance update appropriate for the matched ground truths
                    deltaVariance,
                    sep="\n",
                    logtype="warning",
                )

        # ** Now compute the required probabilities
        # Computations consider all prior information from initialisation and
        # previous batches.

        # The false negative probability (considering all prior information)
        falseNegProb = worker.getNumFalseNegative() / worker.getNumFalseNegativeTrials()

        # The false positive probability (considering all prior information)
        falsePosProb = worker.getNumFalsePositive() / worker.getNumFalsePositiveTrials()

        # The variance (considering all prior information)
        # (note that weights are not included in variance computation)
        variance = worker.getVarianceNumerator() / worker.getNumVarianceTrials()

        return falsePosProb, falseNegProb, variance, worker.workerId

    def computeWorkerSkills_old(self, worker, verbose=False):
        # worker annotations
        annos = worker.getAnnotations()
        annotationIndexLabels = worker.getAnnotationIndexLabels()
        # worker annotations grouped by image
        imageGroupedAnnos = annos.groupby(by="image_id")
        # Images seen by this worker
        workerImages = annos.image_id.unique()
        # *All* annotations for images seen by this worker i.e. even if the
        # worker left it blank.
        workerImageAnnos = worker.annotationStore.annotations.loc[
            pd.Index(self.annoStore.annotations["image_id"]).isin(workerImages)
        ]
        # Number of workers annotating each image
        numWorkersPerImage = imageGroupedAnnos.worker_id.nunique()

        # Workers are assigned a weight to account for the fact that they may
        # be one of many or few that annotated the image. If their assessment
        # agrees/disagrees with a large number of annotators then the
        # increment/decrement to their skill is enhanced.
        multiplicityWeights = ((numWorkersPerImage - 1) / numWorkersPerImage).rename(
            "weights"
        )
        multiplicityWeights = multiplicityWeights.where(multiplicityWeights > 0, 1)

        workerImageAnnos = workerImageAnnos.merge(
            multiplicityWeights, how="left", left_on="image_id", right_index=True
        )  # Now a copy detatched from annoStore

        worker.annotationStore.annotations.loc[
            annotationIndexLabels, "multiplicity_weights"
        ] = workerImageAnnos.loc[annos.index, "weights"]

        # ** False positives
        falsePosProb = (
            (
                worker.getFalsePosPrior()
                * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            )
            + (annos.multiplicity_weights * ~annos.matches_ground_truth).sum()
        ) / (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            + annos.multiplicity_weights.sum()
        )

        # ** False negatives

        # Summed number of ground truth boxes in images seen by this worker
        numGroundTruths = (
            workerImageAnnos.is_ground_truth * workerImageAnnos.weights
        ).sum()
        # Summed number of annotations by this worker that match a ground truth
        numMatchesGroundTruths = (
            annos.multiplicity_weights * annos.matches_ground_truth
        ).sum()
        # The number they missed i.e. number available - number matched
        numMissedGroundTruths = numGroundTruths - numMatchesGroundTruths
        if numMissedGroundTruths < 0:
            self.printToLog(
                "Warning! Apparently negative number of missed ground truths!\n",
                f"{numGroundTruths}, {numMatchesGroundTruths}, {numMissedGroundTruths}",
            )

        # The false negative probability
        falseNegProb = (
            (
                worker.getFalseNegPrior()
                * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
            )
            + numMissedGroundTruths
        ) / (
            numGroundTruths
            + self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
        )

        # ** Variances (note that weights are not included in variance computation)
        variance = (
            (
                worker.getVariancePrior()
                * self.datasetWidePriorParameters["volunteer_skill"][
                    "nInv_chisq_variance"
                ]
            )
            + (
                np.nansum(annos.ground_truth_distance ** 2)
                + (
                    self.statStore.imageStatistics.loc[
                        imageGroupedAnnos.groups.keys(), "variance"
                    ]  # This is why image params must be computed first!
                    * imageGroupedAnnos.matches_ground_truth.sum()
                ).sum()
            )
        ) / (
            numMatchesGroundTruths
            + self.datasetWidePriorParameters["volunteer_skill"]["nInv_chisq_variance"]
        )

        return falsePosProb, falseNegProb, variance

    def computeBatchAssociations(self, init=False):
        # Step 1: Compute facility opening costs
        if init:
            # requiredMultiplictyFraction per worker associated with this image
            self.openCosts = pd.merge(
                self.annoStore.annotations.image_id,
                self.initPhaseParameters["required_multiplicity_fraction"]
                * self.annoStore.getIndexGroupedAnnotations("image_id", rebuild=True)
                .worker_id.nunique()
                .rename("open_cost"),
                how="left",
                left_on="image_id",
                right_index=True,
                suffixes=["", "_openCost"],
            )
        else:
            # -ln(P_fn(box)) per worker associated with this image
            # if the probability of a false negative increases then the cost of
            # opening decreases
            self.openCosts = pd.merge(
                self.annoStore.annotations.image_id,
                -np.log(
                    self.annoStore.getIndexGroupedAnnotations("image_id", rebuild=True)
                    .apply(
                        lambda group: group.groupby(by="worker_id")
                        .false_neg_prob.head(1)
                        .prod()
                    )
                    .rename("open_cost")
                ),
                how="left",
                left_on="image_id",
                right_index=True,
                suffixes=["", "_openCost"],
            )
        self.statStore.setImageOpenCosts(
            np.squeeze(self.openCosts.open_cost.to_numpy()),
            images=self.openCosts.set_index("image_id").index,
        )

        # Step 2: Compute facility-city connection costs
        self.connectionCosts = self.computeConnectionCosts(
            self.distances.to_numpy(),
            self.annoStore.annotations.false_pos_prob.to_numpy(),
            self.annoStore.annotations.false_neg_prob.to_numpy(),
            self.annoStore.annotations.combined_variance.to_numpy(),
            init=init,
            initThreshold=self.initPhaseParameters["required_overlap_fraction"],
        )
        # Step 3: Use facility location algorithm to associate boxes.
        for imageIndex, image in enumerate(self.images):
            # Using groupby will probably not help because data are modified in
            # place
            try:
                if image.haveAnnotations():
                    self.computeAssociations(
                        image,
                        self.openCosts,
                        self.connectionCosts,
                        self.disallowedConnectionMask,
                        verbose=not init,
                    )
                else:
                    self.printToLog(
                        f"\t\t\tNo annotations provided for image {image.imageId}"
                        f" after {image.getNumAnnotations()} views."
                    )
            except KeyError as e:
                self.printToLog(imageIndex, image.imageId, e)
                raise e
        # Step 4: Prune/merge isolated associations
        if not init:
            if self.assocParams["prune_ground_truths"]:
                self.pruneGroundTruths(
                    attemptMerge=self.assocParams["prune_attempt_merge"]
                )
            if self.assocParams["merge_ground_truths"]:
                self.mergeGroundTruths(
                    mergeThreshold=self.assocParams["merge_ground_truth_threshold"]
                )

        # Step 5: Merge associated boxes to determine ground truth coordinates
        self.mergeAssociations()
        # Step 6: Compute the 1-IoU distances between the associated boxes and
        # their corresponding merged ground truths.
        # Delegates computation to numba jit-compiled function
        self.annoStore.annotations.loc[
            self.annotationBoxCoords.index, "ground_truth_distance"
        ] = computeIouWithGroundTruth(
            self.annotationBoxCoords.to_numpy(),
            self.groundTruthBoxCoords[:, :4],
            self.groundTruthIndexSetSizes,
        )

    def computeBoxImageVariance(self, boxData: pd.DataFrame, image: ImageView) -> float:
        """Computes the expected variance for a ground truth box due to local
        image difficulty, based on the corresponding subset of annotation data.

        Parameters
        ----------
        boxData : pandas.DataFrame
            Annotation data associated with this ground truth box.
        image : ImageView
            The image that the ground truth box pertains to.

        Returns
        -------
        float32
            Expected variance for a ground truth box due to local
            image difficulty.
        """
        priorFactor = image.getVarianceNumerator()

        varWeightComplement = 1 - boxData.variance_weighting

        imageVariance = (
            priorFactor
            + (
                np.nansum(
                    varWeightComplement
                    * (
                        boxData.ground_truth_distance ** 2
                        + self.statStore.imageStatistics.loc[
                            image.imageId, "box_variance"
                        ]
                    )
                )
            )
        ) / (varWeightComplement.sum() + image.getNumVarianceTrials())

        return imageVariance

    def computeBoxImageVariance_old(
        self, boxData: pd.DataFrame, image: ImageView
    ) -> float:
        """Computes the expected variance for a ground truth box due to local
        image difficulty, based on the corresponding subset of annotation data.

        Parameters
        ----------
        boxData : pandas.DataFrame
            Annotation data associated with this ground truth box.
        image : ImageView
            The image that the ground truth box pertains to.

        Returns
        -------
        float32
            Expected variance for a ground truth box due to local
            image difficulty.
        """
        priorFactor = (
            image.getVariancePrior()
            * self.datasetWidePriorParameters["image_difficulty"]["nInv_chisq_variance"]
        )

        varWeightComplement = 1 - boxData.variance_weighting

        imageVariance = (
            priorFactor
            + (
                np.nansum(
                    varWeightComplement
                    * (
                        boxData.ground_truth_distance ** 2
                        + self.statStore.imageStatistics.loc[
                            image.imageId, "box_variance"
                        ]
                    )
                )
            )
        ) / (
            varWeightComplement.sum()
            + self.datasetWidePriorParameters["image_difficulty"]["nInv_chisq_variance"]
        )

        return imageVariance

    def computeImageAndLabelParameters(self, image):
        annos = image.getAnnotations()

        unmatchedAnnos = annos[~annos.matches_ground_truth]

        if np.any(annos.matches_ground_truth):
            # Only consider annotations that match a ground truth
            matchingAnnos = annos.loc[annos.matches_ground_truth, :]

            # Image part (image-based variance only)
            associationGroupedMatchingAnnos = matchingAnnos.groupby(by="association")
            groupImageVariances = associationGroupedMatchingAnnos.apply(
                self.computeBoxImageVariance, image
            )
            # Note: Association values refer to a single ground truth within the
            # context of a single image. We are considering annotations for a single
            # image, so indexing based on the "association" column is not an error.
            imageVariance = np.squeeze(
                groupImageVariances.loc[matchingAnnos.association].to_numpy()
            )

            # TODO: could set image variance in statstore for image stats here.

            # Label part (Combination of image and worker variances)
            imageModel = np.exp(
                -0.5 * matchingAnnos.ground_truth_distance ** 2 / imageVariance
            ) / (np.sqrt(2 * np.pi * imageVariance))

            workerModel = np.exp(
                -0.5 * matchingAnnos.ground_truth_distance ** 2 / matchingAnnos.variance
            ) / (np.sqrt(2 * np.pi * matchingAnnos.variance))

            newVarWeights = workerModel / (workerModel + imageModel)

            combinedVariance = ((1 - newVarWeights) * imageVariance) + (
                newVarWeights * matchingAnnos.variance
            )

            self.statStore.setAssociatedBoxes(
                pd.DataFrame(
                    {
                        "image_id": matchingAnnos.image_id,
                        "annotation_id": matchingAnnos.index,
                        "false_pos_prob": matchingAnnos.false_pos_prob,  # same as worker
                        "false_neg_prob": matchingAnnos.false_neg_prob,  # same as worker
                        "image_variance": imageVariance,  # variance based on image stats
                        "combined_variance": combinedVariance,  # variance based on combination of worker and image stats
                        "variance_weighting": newVarWeights,  # The weighting for the combination
                    }
                )
            )
            self.annoStore.annotations.loc[
                matchingAnnos.index, "variance_weighting"
            ] = newVarWeights
            self.annoStore.annotations.loc[
                matchingAnnos.index, "combined_variance"
            ] = combinedVariance
        self.annoStore.annotations.loc[unmatchedAnnos.index, "variance_weighting"] = 1
        # 100% worker weights, since there is no GT box to obtain variance from
        self.annoStore.annotations.loc[
            unmatchedAnnos.index, "combined_variance"
        ] = self.annoStore.annotations.loc[unmatchedAnnos.index, "variance"]

    def computeBatchLikelihoodParameters(self):
        # Step 1: Compute and store the image difficulty variance parameter.
        # The ground truth box variance estimate is computed as a prior
        # estimate divided by the number of ground truth boxes in the
        # image this ground-truth box belongs.
        imageGroupedAnnotations = self.annoStore.annotations.groupby(by="image_id")
        # All boxes in an image have a shared component of variance that
        # decreases as more ground truth boxes are established.
        # TODO: Don't think this needs adapting for multi-batch mode.
        self.statStore.setImageBoxVariances(
            self.datasetWidePriorInitialValues["image_difficulty"]["variance"]
            / np.maximum(
                np.finfo(np.float64).eps, imageGroupedAnnotations.is_ground_truth.sum()
            ),
            imageGroupedAnnotations.groups.keys(),
        )
        # Step 2: Compute and store worker skill parameters and map to annotations
        workerSkills = np.array(
            [self.computeWorkerSkills(worker) for worker in self.workers]
        )
        self.statStore.setWorkerSkills(
            workerSkills[:, :3], workers=workerSkills[:, 3].astype(int)
        )
        self.annoStore.annotations.false_pos_prob = self.statStore.workerStatistics.false_pos_prob.loc[
            self.annoStore.annotations.worker_id
        ].to_numpy()
        self.annoStore.annotations.false_neg_prob = self.statStore.workerStatistics.false_neg_prob.loc[
            self.annoStore.annotations.worker_id
        ].to_numpy()
        self.annoStore.annotations.variance = self.statStore.workerStatistics.variance.loc[
            self.annoStore.annotations.worker_id
        ].to_numpy()
        # Step 3: Compute and store image and label likelihood parameters
        # Note that "label" refers to the set of boxes provided by a specific
        # worker for a specific image.
        for image in self.images:
            self.computeImageAndLabelParameters(image)

    def computeImagePriorLogLikelihood_old(
        self, imageGroundTruthData: pd.Series
    ) -> float:
        """Compute *prior* negative log likelihood for image difficulty parameter
        models, given the *current* parameter estimates.

        The likelihood component is computed in computeLabelLogLikelihood.

        Parameters
        ----------
        imageGroundTruthData : pd.Series
            Accumulated and prior statistical information for a single image.

        Returns
        -------
        float
            Combined negative log-likelihood for image difficulty prior model.

        """
        imageLogLikelihood = (
            (
                (
                    -self.datasetWidePriorParameters["image_difficulty"][
                        "nInv_chisq_variance"
                    ]
                    * self.statStore.imageStatistics.loc[
                        imageGroundTruthData.name, "variance_prior"
                    ]
                )
                / (2 * imageGroundTruthData.image_variance)
            )
            - (
                (
                    1
                    + 0.5
                    * self.datasetWidePriorParameters["image_difficulty"][
                        "nInv_chisq_variance"
                    ]
                )
                * np.log(imageGroundTruthData.image_variance)
            )
        ).sum() / imageGroundTruthData.shape[0]
        return imageLogLikelihood

    def computeImagePriorLogLikelihood(self, imageGroundTruthData: pd.Series) -> float:
        """Compute *prior* negative log likelihood for image difficulty parameter
        models, given the *current* parameter estimates.

        The likelihood component is computed in computeLabelLogLikelihood.

        Parameters
        ----------
        imageGroundTruthData : pd.Series
            Accumulated and prior statistical information for a single image.

        Returns
        -------
        float
            Combined negative log-likelihood for image difficulty prior model.

        """
        stats = self.statStore.imageStatistics.loc[imageGroundTruthData.name]

        imageLogLikelihood = (
            ((-stats.variance_numerator) / (2 * imageGroundTruthData.image_variance))
            - (
                (1 + 0.5 * stats.num_variance_trials)
                * np.log(imageGroundTruthData.image_variance)
            )
        ).sum() / imageGroundTruthData.shape[0]
        return imageLogLikelihood

    def computeWorkerPriorLogLikelihood_old(self, workerData: pd.Series) -> float:
        """Compute *prior* negative log likelihood for worker skill parameter
        models, given the *current* parameter estimates.

        The likelihood component is computed in computeLabelLogLikelihood.

        Parameters
        ----------
        workerData : pandas.Series
            Accumulated and prior statistical information for a single worker.

        Returns
        -------
        float
            Combined negative log-likelihood for worker skill prior model.

        """
        # false positive -log(prior)
        fp = (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            * workerData.false_pos_prob_prior
            - 1
        ) * np.log(workerData.false_pos_prob) + (
            (1 - workerData.false_pos_prob_prior)
            * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            - 1
        ) * np.log(
            1 - workerData.false_pos_prob
        )

        # false negative -log(prior)
        fn = (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
            * workerData.false_neg_prob_prior
            - 1
        ) * np.log(workerData.false_neg_prob) + (
            (1 - workerData.false_neg_prob_prior)
            * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
            - 1
        ) * np.log(
            1 - workerData.false_neg_prob
        )

        # variance log(prior) - not negative.
        var = (
            self.datasetWidePriorParameters["volunteer_skill"]["nInv_chisq_variance"]
            * workerData.variance_prior
            / (2 * workerData.variance)
        ) + (
            1
            + self.datasetWidePriorParameters["volunteer_skill"]["nInv_chisq_variance"]
            / 2
        ) * np.log(
            workerData.variance
        )
        return fp + fn - var

    def computeWorkerPriorLogLikelihood(self, workerData: pd.Series) -> float:
        """Compute *prior* negative log likelihood for worker skill parameter
        models, given the *current* parameter estimates.

        The likelihood component is computed in computeLabelLogLikelihood.

        Parameters
        ----------
        workerData : pandas.Series
            Accumulated and prior statistical information for a single worker.

        Returns
        -------
        float
            Combined negative log-likelihood for worker skill prior model.

        """
        # false positive -log(prior)
        # Note: interpret the complement as "condition not false positive" and
        # not "condition true positive".
        fp = (workerData.num_false_pos - 1) * np.log(workerData.false_pos_prob) + (
            workerData.num_not_false_pos - 1
        ) * np.log(1 - workerData.false_pos_prob)

        # false negative -log(prior)
        # Note: interpret the complement as "condition not false negative" and
        # not "condition true negative".
        fn = (workerData.num_false_neg - 1) * np.log(workerData.false_neg_prob) + (
            workerData.num_not_false_neg - 1
        ) * np.log(1 - workerData.false_neg_prob)

        # variance log(prior) - not negative.
        var = (workerData.variance_numerator / (2 * workerData.variance)) + (
            1 + workerData.num_variance_trials / 2
        ) * np.log(workerData.variance)

        return fp + fn - var

    def computeLabelLogLikelihood(self, imageAnnoData):
        # Contributions for annotations that match a ground truth i.e. true
        # positives
        matchingAnnos = imageAnnoData.loc[imageAnnoData.matches_ground_truth]
        # n_tp*log(gaussian(D, sigma**2) + n_tp*log(1-p_fp))
        truePosLL = (
            -0.5
            * matchingAnnos.ground_truth_distance ** 2
            / matchingAnnos.combined_variance
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(matchingAnnos.combined_variance)
            + np.log(1 - matchingAnnos.false_pos_prob)
        ).sum()

        # Contributions from annotations that do not match a ground truth i.e.
        # false positives
        nonMatchingAnnos = imageAnnoData.loc[~imageAnnoData.matches_ground_truth]
        # TODO: Document this! Method of estimating number of true negatives?
        # If p_fp -> 0 then log(p_fp) -> -inf so allowing more possible ground
        # truths subtracts a positive number and makes p_fp slightly smaller.

        # if many ground truths are plausible then the chance that this worker
        # is the first to annotate it is higher?

        # n_fp*log(p_fp)
        falsePosLL = (
            np.log(nonMatchingAnnos.false_pos_prob)
            - np.log(self.datasetWidePriorParameters["shared"]["max_num_ground_truths"])
        ).sum()

        # Contributions from the incidents when a worker fails to annotate a
        # ground truth i.e. false negatives
        workerGroupFalseNegProbs = matchingAnnos.groupby(by="worker_id").false_neg_prob

        numPossibleMatches = matchingAnnos.association.nunique()
        numWorkerMatches = workerGroupFalseNegProbs.count()
        # n_tp*log(1-p_fn) + n_fn*log(p_fn)
        falseNegLL = (
            np.log(1 - workerGroupFalseNegProbs.head()) * numWorkerMatches
            + np.log(workerGroupFalseNegProbs.head())
            * (numPossibleMatches - numWorkerMatches)
        ).sum()

        return truePosLL + falsePosLL + falseNegLL

    def computeBatchLikelihood(self):
        # Step 1: Compute image negative log-likelihoods.
        # Must process ampty images (with no ground truths) separately.

        # Select image parameters for ground-truth boxes only
        groundTruthAnnos = self.annoStore.annotations.loc[
            self.annoStore.annotations.is_ground_truth
        ]

        imageGroupedGroundTruths = (
            self.statStore.associatedBoxStatistics.reset_index(level=0)
            .loc[groundTruthAnnos.index]
            .groupby(by="image_id")
        )

        self.imagePriorLogLikelihoods = imageGroupedGroundTruths.apply(
            self.computeImagePriorLogLikelihood  ##HERE
        )

        emptyImages = self.annoStore.annotations.loc[
            self.annoStore.annotations.groupby(by="image_id")
            .matches_ground_truth.filter(lambda is_match: not np.any(is_match))
            .index,
            "image_id",
        ]

        self.emptyImagePriorLogLikelihoods = (
            -0.5
            * self.datasetWidePriorParameters["image_difficulty"]["nInv_chisq_variance"]
            - (
                1
                + 0.5
                * self.datasetWidePriorParameters["image_difficulty"][
                    "nInv_chisq_variance"
                ]
            )
            * self.statStore.imageStatistics.loc[emptyImages, "variance_prior"]
        )

        # Step 2: Compute worker negative log-likelihood prior components.
        # Note that we use the cached statistics from initialisation or the
        # previous batch for the prior counts, but the current values for the
        # likelihood parameters.
        priorCountStatColumns = [
            "num_false_pos",
            "num_not_false_pos",
            "num_false_neg",
            "num_not_false_neg",
            "variance_numerator",
            "num_variance_trials",
        ]
        parameterStatColumns = ["variance", "false_pos_prob", "false_neg_prob"]
        workerStats = pd.concat(
            [
                self.statStore.getWorkerStatistics(cached=True).loc[
                    :, priorCountStatColumns
                ],
                self.statStore.getWorkerStatistics(cached=False).loc[
                    :, parameterStatColumns
                ],
            ],
            axis=1,
        )
        self.workerPriorLogLikelihoods = workerStats.apply(
            self.computeWorkerPriorLogLikelihood, axis=1  ##HERE
        )

        # Step 3: Compute label negative log-likelihoods.
        imageGroupedAnnoData = (
            self.annoStore.annotations.loc[
                :,
                [
                    "ground_truth_distance",
                    "is_ground_truth",
                    "matches_ground_truth",
                    "association",
                    "worker_id",
                ],
            ]
            .merge(
                self.statStore.associatedBoxStatistics.reset_index(level=0),
                how="left",
                left_index=True,
                right_index=True,
            )
            .groupby(by="image_id")
        )

        self.labelLogLikelihoods = imageGroupedAnnoData.apply(
            self.computeLabelLogLikelihood
        )

        return (
            self.imagePriorLogLikelihoods.sum()
            + self.emptyImagePriorLogLikelihoods.sum()
            + self.workerPriorLogLikelihoods.sum()
            + self.labelLogLikelihoods.sum()
        )

    def processBatchStep(self, init=False):
        self.printToLog("\tComputing Associations...")
        self.computeBatchAssociations(init=init)
        self.printToLog("\tComputing Likelihood Parameters...")
        self.computeBatchLikelihoodParameters()
        self.printToLog("\tComputing Likelihood Model...")
        return self.computeBatchLikelihood()

    def finaliseBatch(self, discardFinishedImageData=True):
        self.printToLog("Finalising batch")
        # Find finished images.
        finishedImages = self.imageCompletionAssessor(
            self.statStore.imageStatistics, self.imageCompletionParams
        )
        self.printToLog(
            f"\tFound {finishedImages.sum() - self.finishedImageRunningTotal} new finished images."
        )
        self.finishedImageRunningTotal = finishedImages.sum()
        self.printToLog(f"\t{self.finishedImageRunningTotal} finished images in total.")

        finishedImagesIndex = finishedImages.loc[finishedImages].index

        # Update image statistic store to register that images are finished.
        # This will allow future classifications for these images to be
        # discarded in offline mode if desired.
        # Since worker statistics are not reset at the end of a batch, their
        # skill statistics will reflect images that are removed from the
        # pool.
        self.statStore.imageStatistics.loc[
            finishedImagesIndex, ["is_finished", "finish_criterion"]
        ] = [True, "completion_assessor"]

        # For remaining images increment the number of batches they have remained
        # unfinished.
        self.statStore.imageStatistics.loc[
            ~self.statStore.imageStatistics.is_finished, "num_batches_not_finished"
        ] += 1

        # Retire any images that have remained unfinished for more batches than
        # the user-specified threshold.
        maxBatchSelector = ~self.statStore.imageStatistics.is_finished & (
            self.statStore.imageStatistics.num_batches_not_finished
            > self.imageCompletionParams["max_num_batches_not_finished"]
        )
        numStaleImages = maxBatchSelector.sum()
        if numStaleImages > 0:
            self.printToLog(
                f"{numStaleImages} images set finished "
                f"after {self.imageCompletionParams['max_num_batches_not_finished']} "
                "batches without finishing"
            )
        self.statStore.imageStatistics.loc[
            maxBatchSelector, ["is_finished", "finish_criterion"]
        ] = [True, "max_batches_exceeded"]

        # Save all information required to reconstruct clump locations for
        # all finished images. Also save settings for this run.
        if self.savePath is not None:
            finalSavePath = os.path.join(
                self.savePath, f"batch_{self.batchCounter}_final.pkl"
            )
            self.printToLog(f"Saving finalised state to {finalSavePath}...")
            self.saveStores(finalSavePath)
            self.saveSettings(os.path.join(self.savePath, f"settings.pkl"))

        # The prior worker skill values for the next batch should be based only
        # on images that are finished. Unfinished images will be revisited and
        # detected box solutions (and therefore false pos/neg counts) may change.
        self.printToLog(
            "\tSetting worker skill priors using only finished image data..."
        )
        self.statStore.restoreCachedWorkers()
        workerSkills = np.array(
            [
                self.computeWorkerSkills(worker, onlyFinished=True)
                for worker in self.workers
            ]
        )
        self.statStore.setWorkerSkills(
            workerSkills[:, :3], workers=workerSkills[:, 3].astype(int)
        )

        # Remove all annotations for finished images and mark annotations
        # from unfinished images.
        # Note that ImageViews and WorkerViews will be regenerated when new
        # batch initialises.

        if discardFinishedImageData:
            finishedImagesIndex = self.statStore.imageStatistics.loc[
                self.statStore.imageStatistics.is_finished
            ].index
            self.printToLog("\tDiscarding annotations for finished images...")
            self.annoStore.annotations = self.annoStore.annotations.loc[
                ~self.annoStore.annotations.image_id.isin(
                    finishedImagesIndex
                ).to_numpy()
            ].copy()
        self.annoStore.annotations.is_from_earlier_batch = True
        # Explicitly clear statistic store caches
        self.statStore.clearCache()

    def computeImageExpNumFalseNegative(self, imageAnnotations, grouper):
        # Can only compute false negative probability using data for this image
        # if some boxes were unassociated!
        if (imageAnnotations.association < 0).any():
            cities = np.flatnonzero(~imageAnnotations.matches_ground_truth)
            facilities = np.flatnonzero(~imageAnnotations.is_ground_truth)

            fcPairs = np.array(list(itertools.product(facilities, cities)))
            try:
                # Boxes canot connect to themselves.
                fcPairs = fcPairs[fcPairs[:, 0] != fcPairs[:, 1]].T
            except IndexError as e:
                self.printToLog(e, fcPairs.shape, cities.shape, facilities.shape)

            imageAnnotationIndices = grouper.indices[imageAnnotations.name]
            # Add 1 since costs and connection mask *do* have rows/columns for the dummy
            selector = np.ix_(
                [0] + (imageAnnotationIndices + 1).tolist(),
                [0] + (imageAnnotationIndices + 1).tolist(),
            )

            openCostSubset = np.nan_to_num(
                self.openCosts.to_numpy()[imageAnnotationIndices, 1].tolist()
            )

            # Includes dummy
            connectionCostSubset = self.connectionCosts[selector]

            fl = FacilityLocation(
                openCostSubset,
                connectionCostSubset[1:, 1:],  # excludes dummy
                self.disallowedConnectionMask.to_numpy()[selector][1:, 1:].astype(bool),
                logging.WARNING,
            )
            fl.solve(fcPairs=fcPairs, haveDummy=False)

            # open facilities
            facilities = np.flatnonzero(fl.fData.data[:, FData.isOpen])
            openFacCosts = openCostSubset[facilities]
            # cities connected to open facilities are True
            cityConnectedIndicator = np.flatnonzero(
                np.isin(fl.cData.data[:, CData.facility], facilities)
            )
            cityConnectCosts = connectionCostSubset[1:, 1:][  # excluding dummy
                np.ix_(facilities, cityConnectedIndicator)
            ]
            # Cost for assigning TP label i.e cost of opening + summed cost
            # of connections.
            truePosConnectCosts = np.nansum(cityConnectCosts, axis=1) + openFacCosts

            # Cost for retaining original FP label (connecting cities to dummy)
            # i.e. sum of connection costs to dummy facility (open cost is 0)

            # which cities want to connect to all open facilities?
            cityConnectedIndicator2d = np.nan_to_num(cityConnectCosts) > 0
            # How much does it cost them to connect to the dummy instead?
            cityConnectCostsDummy = connectionCostSubset[:, 1:][
                0, np.isin(fl.cData.data[:, CData.facility], facilities)
            ]
            # project costs along facility axis and sum along city axis
            falsePosConnectCosts = (
                cityConnectedIndicator2d * cityConnectCostsDummy
            ).sum(axis=1)

            maxConnectCost = np.maximum(truePosConnectCosts, falsePosConnectCosts)

            expectedNumFalseNeg = (
                np.exp(-(truePosConnectCosts - maxConnectCost))
                / (
                    np.exp(-(truePosConnectCosts - maxConnectCost))
                    + np.exp(-(falsePosConnectCosts - maxConnectCost))
                )
            ).sum()
            return expectedNumFalseNeg

        return 0

    def computeExpNumFalseNegative(self):
        # Compute IoU distance between ground truths and all boxes in this batch
        self.bigBBoxDistances = computeIouDistancesAsymm(
            self.groundTruthBoxCoords[:, 4:],
            self.bigBBoxSet.loc[:, BoxAggregator.normedBoxCoordCols].to_numpy(),
        )
        # Find all boxes in the global set that do not intersect with a particular
        # ground truth box.
        # NOTE: Previously raised warning because bigBBoxDistances intentionally
        # contains NaN values (so comparison would always return false)
        self.bigBBoxMisses = (
            np.nan_to_num(self.bigBBoxDistances, 2)
            > self.initPhaseParameters["required_overlap_fraction"]
        )

        # The global set approximates the probability that any position in any
        # image contains a target object e.g. all likely targets may be
        # concentrated in the centre of the image.
        #
        # Every global box that doesn't intersect with a ground truth box for
        # this image increases the probability that there is a real box missing
        # from the current ground truth estimate.
        #
        # If the cost of identifying the ground truth box as a target was high,
        # then this reduces the chance that the missed intersection implies a
        # missed target.
        #
        # Recall that the cost is the sum of negated log false negative
        # probabilities over all workers who annotated the image corresponding
        # to a particluar box i.e. a higher cost implies many workers with a
        # low false negative probability annotated the image.
        imageOpenCosts = self.openCosts.groupby(by="image_id").first()
        self.bigBBoxMissContributions = (
            pd.DataFrame(self.bigBBoxMisses, columns=self.bigBBoxSet.index)
            .groupby(by=self.groundTruthBoxImageIds)
            .apply(
                lambda grp, fpp, oc: np.sum(
                    np.all(grp, axis=0) * (fpp) * np.exp(-oc.loc[grp.name, "open_cost"])
                ),
                self.bigBBoxSet.random_coincidence_prob,
                imageOpenCosts,
            )
        )

        # Now note that any image with no ground truths also incurs a risk of
        # false negatives, but since no ground truths can possibly intersect
        # then the risk is simply the sum of all random coincidence probabilities
        # weighted by the negative exponential of the open cost for that image.
        # Note this also includes images that have no annotations at all.
        noAssocImageIds = pd.Index(self.annoStore.annotations.image_id).difference(
            self.groundTruthBoxImageIds
        )
        self.noAssocBigBBoxMissContributions = self.bigBBoxSet.random_coincidence_prob.sum() * np.exp(
            -imageOpenCosts.loc[noAssocImageIds, "open_cost"]
        )

        imageGroupedAnnotations = self.annoStore.annotations.groupby(by="image_id")

        # Compute expected false negatives based on annotations for particular
        # images.
        expNumFalseNegative = imageGroupedAnnotations.apply(
            self.computeImageExpNumFalseNegative, imageGroupedAnnotations
        )

        # save image-only contributions
        self.expNumFalseNegativeImage = expNumFalseNegative.copy()

        # Combine image-specific false negative count expectation with global
        # expectations.
        expNumFalseNegative.loc[
            self.bigBBoxMissContributions.index
        ] += self.bigBBoxMissContributions
        expNumFalseNegative.loc[noAssocImageIds] += self.noAssocBigBBoxMissContributions

        self.statStore.setImageExpNumFalseNegative(
            expNumFalseNegative, expNumFalseNegative.index
        )

    def computeExpNumFalsePositive(self, singleBoxFilterThreshold=None):
        # Get a reference to all ground annotations that match a ground truth.
        matchingAnnos = self.annoStore.annotations.loc[
            self.annoStore.annotations.matches_ground_truth
        ]

        # Compute a set of workers who matched each ground truth
        matchingWorkers = (
            matchingAnnos.groupby(by=["image_id", "association"])
            .worker_id.apply(set)
            .reset_index(level=1)
        )

        # Compute a set of workers who saw the image and had an opportunity
        # to mark the ground truth
        possibleWorkers = self.annoStore.annotations.groupby(
            by="image_id"
        ).worker_id.apply(set)

        combinedWorkers = matchingWorkers.merge(
            possibleWorkers,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=["_matched", "_possible"],
        )

        # Compute a set of workers who missed each ground truth
        combinedWorkers["worker_id_missed"] = (
            combinedWorkers.worker_id_possible - combinedWorkers.worker_id_matched
        )
        # Retrieve the false negative probabilities for workers that missed each
        # ground truth
        combinedWorkers[
            "false_neg_prob_missed"
        ] = combinedWorkers.worker_id_missed.apply(
            lambda ids: self.statStore.workerStatistics.loc[
                ids, "false_neg_prob"
            ].to_numpy()
        )
        # Compute the complement of the probabilities that were just retrieved
        # i.e. the true negative probability for workers who missed.
        combinedWorkers[
            "false_neg_prob_missed_complement"
        ] = combinedWorkers.false_neg_prob_missed.apply(lambda probs: 1 - probs)

        # Retrieve the false positive probabilities for workers that matched each ground truth
        combinedWorkers[
            "false_pos_prob_matched"
        ] = combinedWorkers.worker_id_matched.apply(
            lambda ids: self.statStore.workerStatistics.loc[
                ids, "false_pos_prob"
            ].to_numpy()
        )
        # Compute the complement of the probabilities that were just retrieved
        # i.e. the true positive probability for workers who matched.
        combinedWorkers[
            "false_pos_prob_matched_complement"
        ] = combinedWorkers.false_pos_prob_matched.apply(lambda probs: 1 - probs)

        # Compute the probability that each ground truth is a true positive
        # This is the product of the false negative probability for all workers that missed
        # and the complement of the false positive probability i.e. the true positive
        # probability for all workers that matched.
        combinedWorkers[
            "assoc_true_pos_prob"
        ] = combinedWorkers.false_pos_prob_matched_complement.apply(
            np.product
        ) * combinedWorkers.false_neg_prob_missed.apply(
            np.product
        )

        # Compute the probability that this ground truth is a false positive
        # This is the product of the false postive probability for all workers that matched
        # and the complement of the false negative probability i.e. the true negative
        # probability for all workers that missed.
        combinedWorkers[
            "assoc_false_pos_prob"
        ] = combinedWorkers.false_neg_prob_missed_complement.apply(
            np.product
        ) * combinedWorkers.false_pos_prob_matched.apply(
            np.product
        )
        # Compute the contribution of each ground truth to the expected count of false
        # positives for its associated image.
        # Computed as the false pos probability divided by the probability that the
        # ground truth is identified.
        combinedWorkers["false_pos_prob"] = combinedWorkers.assoc_false_pos_prob / (
            combinedWorkers.assoc_false_pos_prob + combinedWorkers.assoc_true_pos_prob
        )

        # Compute the expected number of false positives for each image by summing
        # the contributions of each identified ground truth.
        filteredCombinedWorkers = combinedWorkers
        if singleBoxFilterThreshold is not None:
            filteredCombinedWorkers = combinedWorkers.loc[
                combinedWorkers.false_pos_prob < singleBoxFilterThreshold
            ]

        expNumFalsePositive = (
            filteredCombinedWorkers.reset_index()
            .groupby(by=["image_id"])
            .false_pos_prob.sum()
            # .reindex(self.annoStore.annotations.image_id.unique())
            # .fillna(0)
        )

        self.statStore.setImageExpNumFalsePositive(
            expNumFalsePositive, expNumFalsePositive.index
        )

        return combinedWorkers.loc[:, ["association", "false_pos_prob"]].set_index(
            "association", append=True
        )

    def computeExpNumInaccurate(self):
        """
        To estimate the number of ground truth boxes with a variance exceeding some threshold
        distance model the ground truth position as a sample from a Gaussian distribution
        with zero mean and variance $\sigma^{2}$ given by the sum of combined (worker + image) variances
        for each box associated with the ground truth.

        In this case, the probability that the error associated with the ground truth box
        position exceeds $\pm \delta$ is given by $1-\mathrm{erfc}\left(\frac{\delta}{\sqrt{2\sigma^{2}}}\right)$.

        Note that the variances are expressed in 1-IoU coordinates and are therefore bounded in the range $[0,1]$
        """
        # Step 0: Get a reference to all ground annotations that match a ground truth.
        matchingAnnos = self.annoStore.annotations.loc[
            self.annoStore.annotations.matches_ground_truth
        ]

        # Step 1: Extract the combined variances for the annotations that match a ground truth
        matchingGroundTruthStats = (
            self.statStore.associatedBoxStatistics.reset_index(level=0)
            .loc[matchingAnnos.index]
            .set_index("image_id", append=True)
        )
        combinedVariances = matchingGroundTruthStats.combined_variance

        # Step 2: Group the extracted variances according to the image and ground truth association
        # within that image to which they pertain and sum them to obtain an estimate of the overall
        # variance of the ground truth box.
        summedCombinedInverseVariances = (
            (1 / combinedVariances)
            .groupby(
                by=[
                    self.annoStore.annotations.loc[
                        matchingAnnos.index, "image_id"
                    ].to_numpy(),
                    self.annoStore.annotations.loc[
                        matchingAnnos.index, "association"
                    ].to_numpy(),
                ]
            )
            .sum()
        )

        # Step 3: Compute the probability that each the positional error on each ground truth
        # exceeds a threshold overlap fraction. Note that smaller values of required_accuracy
        # imply a tighter cluster of associated boxes.
        groundTruthInaccurateProbs = sp.special.erfc(
            self.lossParams["required_accuracy"]
            * np.sqrt(0.5 * summedCombinedInverseVariances)
        )

        # Step 4: Compute the expected number of ground truth boxes in each image with positional
        # error that exceeds the threshold overlap fraction by summing the probabilities that each
        # ground truth box associated with the image does so.
        expectedNumInaccurate = groundTruthInaccurateProbs.groupby(level=0).sum()
        self.statStore.setImageExpNumInaccurate(
            expectedNumInaccurate, expectedNumInaccurate.index
        )

        groundTruthInaccurateProbs = groundTruthInaccurateProbs.rename("inaccurate_prob").to_frame()
        groundTruthInaccurateProbs.index.set_names(["image_id", "association"])
        return groundTruthInaccurateProbs

    def computeRisks(self):
        # Step 6: Compute risks for all images.
        self.printToLog("\tComputing risks...")
        # Step 6a: Compute expected number of false negatives for all images.
        self.printToLog("\t\tComputing per-image expected false negative counts...")
        self.computeExpNumFalseNegative()

        # Step 6b: Compute expected number of false positives for all images.
        self.printToLog("\t\tComputing per-image expected false positive counts...")
        gtFalsePosProbs = self.computeExpNumFalsePositive(
            singleBoxFilterThreshold=self.lossParams["single_box_filter_threshold"]
        )

        # Step 6c: Compute expected variance of (assumed) true positive annotations.
        self.printToLog(
            "\t\tComputing per-image expected inaccurate positive counts..."
        )
        gtInaccurateProbs = self.computeExpNumInaccurate()
        # self.printToLog(gtInaccurateProbs, gtFalsePosProbs, sep="\n\n")

        # Step 6d: Compute individual ground truth box risks
        groundTruthRisks = self.lossParams["false_pos_loss"] * (
            gtFalsePosProbs.false_pos_prob
            + (
                gtInaccurateProbs.inaccurate_prob
                # TODO: What is this weighting all about?
                * (
                    self.lossParams["false_neg_loss"]
                    + (
                        1 - gtFalsePosProbs.false_pos_prob
                    )  # Additional weight if the box is likely a true positive
                    * self.lossParams["false_pos_loss"]
                )
            )
        ).rename("risk")

        # Step 6e: Save individual ground truth box statistics
        print("***", gtFalsePosProbs, gtInaccurateProbs, groundTruthRisks, sep="\n\n")
        self.statStore.setGroundTruths(
            pd.concat(
                [gtFalsePosProbs, gtInaccurateProbs, groundTruthRisks], axis=1
            ).reset_index()
        )

        # Step 6f: Compute per image risks
        filteredGroundTruthRisks = groundTruthRisks
        if self.lossParams["single_box_filter_threshold"] is not None:
            filteredGroundTruthRisks = groundTruthRisks.loc[
                gtFalsePosProbs.false_pos_prob
                < self.lossParams["single_box_filter_threshold"]
            ]

        imageRisks = (
            groundTruthRisks.groupby(level=0)
            .sum()
            .reindex_like(self.statStore.imageStatistics)
            .fillna(0)
        )

        imageRisks += (
            self.statStore.imageStatistics.loc[:, "expected_num_false_neg"]
            * self.lossParams["false_neg_loss"]
        )
        # Step 6g: Save per image risks
        self.statStore.setImageRisk(imageRisks, imageRisks.index)

    def processBatch(
        self, dataBatch, stopEarlyAfterStep=None, discardFinishedImageData=True
    ):
        self.setupNewBatch(dataBatch)

        # ** Once-per-batch computations
        # Step 1: Compute 1-IOU distances between all annotations in normalised coordinates
        self.printToLog("Computing IoU distances...")
        self.distances = self.imageProcessor.getIouDistances(
            store=self.annoStore, shuffle=False
        )
        # Step 2: Compute initial disallowed connection mask for all annotations
        self.printToLog("Computing initial disallowed connection mask...")
        exclusiveIndexSetSizes = (
            self.annoStore.getMultiIndexGroupedAnnotations(rebuild=True)
            .count()
            .iloc[:, 0]
            .values
        )
        exclusiveIndices = np.concatenate(
            list(self.annoStore.getMultiIndexGroupedAnnotations().indices.values())
        )

        # Actual computation is delegated to a numba jit-compiled function
        self.disallowedConnectionMask = pd.DataFrame(
            computeDisallowedConnectionMask(
                exclusiveIndices, exclusiveIndexSetSizes, self.distances.shape
            ),
            index=[-1] + self.annoStore.annotations.index.tolist(),
            columns=[-1] + self.annoStore.annotations.index.tolist(),
        )
        # Step 3: Compute global box overlap statistics for later use computing
        # risk.
        self.printToLog("Computing global box overlap statistics...")
        # Note this is *not* inefficient since we are just shuffling previously
        # computed distances
        shuffledDistances, shuffledIndex = self.imageProcessor.getIouDistances(
            store=None, shuffle=True
        )
        validIndices = self.annoStore.annotations[
            ~self.annoStore.annotations["empty"]
        ].index
        overlaps = (
            shuffledDistances.loc[validIndices, validIndices]
            < self.initPhaseParameters["required_overlap_fraction"]
        )

        # Search for groups of boxes that obverlap in normalised image
        # coordinate space.
        randomOverlapCounts = collections.OrderedDict({})

        for newBox in overlaps.index:
            overlapFound = False
            for knownBox in randomOverlapCounts.keys():
                if overlaps.loc[knownBox, newBox]:
                    # If the box being considered overlaps with one that has
                    # already been processed then increment the overlap
                    # count of the *previous* box.
                    randomOverlapCounts[knownBox] += 1
                    overlapFound = True
                    break
            # After comparing with all previous boxes.
            else:
                if not overlapFound:
                    # If the box being considered does not overlap with one
                    # that has already been processed then increment add this
                    # box to the list for future comparisons.
                    randomOverlapCounts[newBox] = 0

        randomOverlapCounts = pd.Series(randomOverlapCounts)

        # The random false positive probability is the number of random overlaps
        # divided by the number of distinct annotations i.e. unique, non-empty
        # worker-image pairs.
        numAnnotations = (
            self.annoStore.annotations.loc[~self.annoStore.annotations["empty"]]
            .groupby(by=["worker_id", "image_id"])
            .ngroups
        )
        randomCoincidenceProbs = (randomOverlapCounts / numAnnotations).rename(
            "random_coincidence_prob"
        )

        self.bigBBoxSet = self.annoStore.annotations.loc[
            validIndices,
            [
                "x1_normed",
                "x2_normed",
                "y1_normed",
                "y2_normed",
                "worker_id",
                "image_id",
            ],
        ].merge(randomCoincidenceProbs, how="left", left_index=True, right_index=True)

        # Step 4:
        # ** Initial batch processing
        self.printToLog("Initial batch processing...")
        self.statStore.cacheWorkers()
        self.statStore.cacheImages()
        self.statStore.resetAssociatedBoxes()
        self.statStore.resetGroundTruths()

        self.processBatchStep(init=True)
        if self.computeStepwiseRisks:
            self.computeRisks()
        if self.savePath is not None:
            self.saveStores(
                os.path.join(self.savePath, f"batch_{self.batchCounter}_init.pkl")
            )

        if stopEarlyAfterStep == "init":
            self.printToLog(f"Stopping early after init.")
            return

        self.statStore.restoreCachedWorkers()
        self.statStore.restoreCachedImages()

        # Step 5:
        # ** Expectation maximisation
        currentLikelihood = np.inf
        previousLikelihood = np.inf
        for batchStep in range(self.maxBatchIterations):
            self.printToLog(f"Batch processing iteration {batchStep}...")
            self.statStore.resetAssociatedBoxes()
            self.statStore.resetGroundTruths()
            self.statStore.cacheWorkers()
            self.statStore.cacheImages()

            currentLikelihood = self.processBatchStep(init=False)
            # Store record of likelihood value for this iteration
            self.likelihoods.append(currentLikelihood)
            if self.computeStepwiseRisks:
                self.computeRisks()
            if self.savePath is not None:
                stepSavePath = os.path.join(
                    self.savePath, f"batch_{self.batchCounter}_step_{batchStep}.pkl"
                )
                self.saveStores(stepSavePath)
            if currentLikelihood >= previousLikelihood:
                self.printToLog(f"Converged @ {currentLikelihood}")
                break
            elif stopEarlyAfterStep == batchStep:
                self.printToLog(
                    f"Stopping early after step {batchStep}. Likelihood @ {currentLikelihood}"
                )
                break
            else:
                self.printToLog(
                    f"Not Converged @ {currentLikelihood} (Previous: {previousLikelihood})"
                )
                previousLikelihood = currentLikelihood

                if batchStep < self.maxBatchIterations - 1:
                    # Restore cached worker/image statistics
                    self.statStore.restoreCachedWorkers()
                    self.statStore.restoreCachedImages()

        if not self.computeStepwiseRisks:
            self.computeRisks()

        # Determine finished images, cleanup and persistence operations.
        self.finaliseBatch(discardFinishedImageData=discardFinishedImageData)

    def saveStores(self, savePath):
        self.printToLog(f"Saving state to {savePath}...")
        saveable = dict(
            annotations=self.annoStore.getSaveable(),
            statistics=self.statStore.getSaveable(),
        )
        with open(savePath, mode="wb") as saveFile:
            pickle.dump(saveable, saveFile)

    def saveSettings(self, savePath):
        self.printToLog(f"Saving settings to {savePath}...")
        saveable = dict(
            datasetWidePriorInitialValues=self.datasetWidePriorInitialValues,
            datasetWidePriorParameters=self.datasetWidePriorParameters,
            initPhaseParameters=self.initPhaseParameters,
            filterParams=self.filterParams,
            lossParams=self.lossParams,
        )
        with open(savePath, mode="wb") as saveFile:
            pickle.dump(saveable, saveFile)

    def loadStores(self, loadPath):
        with open(loadPath, mode="rb") as loadFile:
            loaded = pickle.load(loadFile)
            self.annoStore = AnnotationStore.fromSaveable(loaded["annotations"])
            self.statStore = StatisticStore.fromSaveable(loaded["statistics"])
