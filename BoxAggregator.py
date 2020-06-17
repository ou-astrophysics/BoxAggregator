import numpy as np
import pandas as pd
import scipy as sp

import numba
import logging
import itertools

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
        for right in range(left + 1, distances.shape[1]):
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
        shared=dict(variance=0.5),
    )

    defaultDatasetWidePriorParameters = dict(
        volunteer_skill=dict(
            nBeta_false_pos=500, nBeta_false_neg=10, nInv_chisq_variance=10
        ),
        image_difficulty=dict(nInv_chisq_variance=10),
        shared=dict(max_num_ground_truths=20),
    )

    defaultInitPhaseParams = dict(
        required_multiplicity_fraction=0.1, required_overlap_fraction=0.6
    )

    defaultFilterParams = dict(min_images_per_worker=10, min_workers_per_image=5)

    defaultLossParams = dict(false_pos_loss=1, false_neg_loss=1, required_accuracy=0.2)

    boxCoordCols = ["x1", "x2", "y1", "y2"]
    normedBoxCoordCols = ["x1_normed", "x2_normed", "y1_normed", "y2_normed"]

    def __init__(
        self,
        dataBatch,
        filterInputData=True,
        filterParams=None,
        initPhaseParams=None,
        lossParams=None,
        datasetWidePriorInitialValues=None,
        datasetWidePriorParameters=None,
    ):

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

        self.filterParams = BoxAggregator.defaultFilterParams
        if filterParams is not None:
            self.filterParams.update(filterParams)

        self.lossParams = BoxAggregator.defaultLossParams
        if lossParams is not None:
            self.lossParams.update(lossParams)

        self.dataBatch = (
            self.filterInputData(dataBatch, filterParams)
            if filterInputData
            else dataBatch
        )

        # Class that implements image-specific computations
        self.imageProcessor = ImageProcessor()

        self.setup()

        self.processBatch()

    def filterInputData(self, dataBatch, filterParams):
        return (
            dataBatch.groupby(by=["worker_id"])
            .filter(
                lambda x: x.image_id.unique().size
                > filterParams["min_images_per_worker"]
            )
            .groupby(by=["image_id"])
            .filter(
                lambda x: x.worker_id.unique().size
                > filterParams["min_workers_per_image"]
            )
        ).reset_index(drop=True)

    def setup(self):
        self.annoStore = AnnotationStore()
        self.statStore = StatisticStore()

        self.annoStore.addAnnotations(
            self.dataBatch, self.datasetWidePriorInitialValues
        )
        self.images = self.annoStore.generateViews(
            "image_id", ImageView, self.statStore
        )
        self.workers = self.annoStore.generateViews(
            "worker_id", WorkerView, self.statStore
        )

        self.statStore.addAnnotations(
            self.annoStore, self.datasetWidePriorInitialValues
        )

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
        annotationIndices = image.getAnnotationIndices()
        selector = np.ix_(
            [0] + (annotationIndices + 1).tolist(),
            [0] + (annotationIndices + 1).tolist(),
        )
        fl = FacilityLocation(
            np.nan_to_num([0] + openCosts.values[annotationIndices, 1].tolist()),
            connectionCosts[selector],
            disallowedConnectionMask.to_numpy()[selector].astype(bool),
            logging.WARNING,
        )
        fl.solve(haveDummy=True)

        image.annotationStore.annotations.loc[
            annotationIndices, "is_ground_truth"
        ] = fl.fData.data[1:, FData.isOpen].astype(bool)

        image.annotationStore.annotations.loc[annotationIndices, "association"] = (
            fl.cData.data[1:, CData.facility] - 1
        )

        gtAnnotationIndices = np.flatnonzero(
            image.annotationStore.annotations.loc[annotationIndices, "is_ground_truth"]
        )

        image.annotationStore.annotations.loc[
            annotationIndices[gtAnnotationIndices], "association"
        ] = gtAnnotationIndices

        # Require that the associated facility is open and is not the dummy.
        image.annotationStore.annotations.loc[
            annotationIndices, "matches_ground_truth"
        ] = (
            image.annotationStore.annotations.loc[annotationIndices, "association"] >= 0
        )

    def mergeAssociations(self):
        ## Compute average bounding boxes from all boxes associated with a particular ground truth.

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

    def computeWorkerSkills(self, worker, verbose=False):
        # worker annotations
        annos = worker.getAnnotations()
        # worker annotations grouped by image
        imageGroupedAnnos = annos.groupby(by="image_id")
        # Images seen by this worker
        workerImages = annos.image_id.unique()
        # *All* annotations for images seen by this worker
        workerImageAnnos = worker.annotationStore.annotations.loc[
            pd.Index(self.annoStore.annotations["image_id"]).isin(workerImages)
        ]
        # Number of workers annotating each image
        numWorkersPerImage = imageGroupedAnnos.worker_id.nunique()

        # Workers are assigned a weight to account for the fact that they may
        # be one of many or few that annotated the image. If their assessment
        # agrees/disagrees with a large number of annotators then the
        # increment/decrement to their skill is enhanced.
        weights = ((numWorkersPerImage - 1) / numWorkersPerImage).rename("weights")
        weights = weights.where(weights > 0, 1)

        workerImageAnnos = workerImageAnnos.merge(
            weights, how="left", left_on="image_id", right_index=True
        )

        annos["weights"] = workerImageAnnos.loc[annos.index, "weights"]

        # ** False positives
        falsePosProb = (
            (
                worker.getFalsePosPrior()
                * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            )
            + (annos.weights * ~annos.matches_ground_truth).sum()
        ) / (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            + annos.weights.sum()
        )

        # ** False negatives

        # Summed number of ground truth boxes in images seen by this worker
        numGroundTruths = (
            workerImageAnnos.is_ground_truth * workerImageAnnos.weights
        ).sum()
        # Summed number of annotations by this worker that match a ground truth
        numMatchesGroundTruths = (annos.weights * annos.matches_ground_truth).sum()
        # The number they missed i.e. number available - number matched
        numMissedGroundTruths = numGroundTruths - numMatchesGroundTruths
        if numMissedGroundTruths < 0:
            print(
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
                * self.annoStore.getIndexGroupedAnnotations(
                    "image_id", rebuild=True
                ).worker_id.nunique(),
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
                    .rename("false_neg_prob")
                ),
                how="left",
                left_on="image_id",
                right_index=True,
                suffixes=["", "_openCost"],
            )
        # Step 2: Compute facility-city connection costs
        self.connectionCosts = self.computeConnectionCosts(
            self.distances.to_numpy(),
            self.annoStore.annotations.false_pos_prob.values,
            self.annoStore.annotations.false_neg_prob.values,
            self.annoStore.annotations.combined_variance.values,
            init=init,
            initThreshold=self.initPhaseParameters["required_overlap_fraction"],
        )
        # Step 3: Use facility location algorithm to associate boxes.
        for image in self.images:
            # Using groupby will probably not help because data are modified in
            # place
            self.computeAssociations(
                image,
                self.openCosts,
                self.connectionCosts,
                self.disallowedConnectionMask,
            )
        # Step 4: Merge associated boxes to determine ground truth coordinates
        self.mergeAssociations()
        # Step 5: Compute the 1-IoU distances between the associated boxes and
        # their corresponding merged ground truths.
        # Delegates computation to numba jit-compiled function
        self.annoStore.annotations.loc[
            self.annotationBoxCoords.index, "ground_truth_distance"
        ] = computeIouWithGroundTruth(
            self.annotationBoxCoords.to_numpy(),
            self.groundTruthBoxCoords[:, :4],
            self.groundTruthIndexSetSizes,
        )

    def computeImageAndLabelParameters(self, image):
        annos = image.getAnnotations()
        # annoIndices = image.getAnnotationIndices()

        # Only consider annotations that match a ground truth
        matchingAnnos = annos.loc[annos.matches_ground_truth, :]
        # numMatchingAnnos = matchingAnnos.shape[0]

        unmatchedAnnos = annos[~annos.matches_ground_truth]

        # Image part (image-based variance only)
        priorFactor = (
            image.getVariancePrior()
            * self.datasetWidePriorParameters["image_difficulty"]["nInv_chisq_variance"]
        )

        varWeightComplement = 1 - matchingAnnos.variance_weighting

        imageVariance = (
            priorFactor
            + (
                np.nansum(
                    varWeightComplement
                    * (
                        matchingAnnos.ground_truth_distance ** 2
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
        self.statStore.setImageBoxVariances(
            self.datasetWidePriorInitialValues["image_difficulty"]["variance"]
            / np.maximum(
                np.finfo(np.float64).eps, imageGroupedAnnotations.is_ground_truth.sum()
            ),
            imageGroupedAnnotations.groups.keys(),
        )
        # Step 2: Compute and store worker skill parameters and map to annotations
        self.statStore.setWorkerSkills(
            np.array([self.computeWorkerSkills(worker) for worker in self.workers])
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

    def computeImageLogLikelihood(self, imageGroundTruthData):
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

    def computeWorkerLogLikelihood(self, workerData):
        # false positive
        fp = (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            * workerData.false_pos_prob
            - 1
        ) * np.log(workerData.false_pos_prob) + (
            (1 - workerData.false_pos_prob)
            * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_pos"]
            - 1
        ) * np.log(
            1 - workerData.false_pos_prob
        )

        # false negative
        fn = (
            self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
            * workerData.false_neg_prob
            - 1
        ) * np.log(workerData.false_neg_prob) + (
            (1 - workerData.false_neg_prob)
            * self.datasetWidePriorParameters["volunteer_skill"]["nBeta_false_neg"]
            - 1
        ) * np.log(
            1 - workerData.false_neg_prob
        )

        # variance
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

    def computeLabelLogLikelihood(self, imageAnnoData):
        matchingAnnos = imageAnnoData.loc[imageAnnoData.matches_ground_truth]

        truePosLL = (
            -0.5
            * matchingAnnos.ground_truth_distance ** 2
            / matchingAnnos.combined_variance
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(matchingAnnos.combined_variance)
            + np.log(1 - matchingAnnos.false_pos_prob)
        ).sum()

        nonMatchingAnnos = imageAnnoData.loc[~imageAnnoData.matches_ground_truth]
        falsePosLL = (
            np.log(nonMatchingAnnos.false_pos_prob)
            - np.log(self.datasetWidePriorParameters["shared"]["max_num_ground_truths"])
        ).sum()

        # Question: What if a worker sees an image, but annotates nothing? Should affect FN.
        workerGroupFalseNegProbs = matchingAnnos.groupby(by="worker_id").false_neg_prob

        # Did a worker supply a box that matched a ground truth
        numPossibleMatches = matchingAnnos.association.nunique()
        numWorkerMatches = workerGroupFalseNegProbs.count()
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

        self.imageLogLikelihoods = imageGroupedGroundTruths.apply(
            self.computeImageLogLikelihood
        )

        emptyImages = self.annoStore.annotations.loc[
            self.annoStore.annotations.groupby(by="image_id")
            .matches_ground_truth.filter(lambda is_match: not np.any(is_match))
            .index,
            "image_id",
        ]

        self.emptyImageLogLikelihoods = (
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

        # Step 2: Compute worker negative log-likelihoods.
        self.workerLogLikelihoods = self.statStore.workerStatistics.apply(
            self.computeWorkerLogLikelihood, axis=1
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
            self.imageLogLikelihoods.sum()
            + self.emptyImageLogLikelihoods.sum()
            + self.workerLogLikelihoods.sum()
            + self.labelLogLikelihoods.sum()
        )

    def processBatchStep(self, init=False):
        print("\tComputing Associations...")
        self.computeBatchAssociations(init=init)
        print("\tComputing Likelihood Parameters...")
        self.computeBatchLikelihoodParameters()
        print("\tComputing Likelihood Model...")
        return self.computeBatchLikelihood()

    def computeImageExpNumFalseNegative(self, imageAnnotations):
        # Can only compute false negative probability using data for this image
        # if some boxes were unassociated!
        if (imageAnnotations.association < 0).any():
            cities = np.flatnonzero(~imageAnnotations.matches_ground_truth)
            facilities = np.flatnonzero(~imageAnnotations.is_ground_truth)

            fcPairs = np.array(list(itertools.product(facilities, cities)))
            fcPairs = fcPairs[fcPairs[:, 0] != fcPairs[:, 1]].T

            # Add 1 since costs and connection mask *do* have rows/columns for the dummy
            selector = np.ix_(
                [0] + (imageAnnotations.index.values + 1).tolist(),
                [0] + (imageAnnotations.index.values + 1).tolist(),
            )

            openCostSubset = np.nan_to_num(
                self.openCosts.values[imageAnnotations.index.values, 1].tolist()
            )

            # Includes dummy
            connectionCostSubset = self.connectionCosts[selector]

            fl = FacilityLocation(
                openCostSubset,
                connectionCostSubset[1:, 1:],  # excludes dummy
                self.disallowedConnectionMask.to_numpy()[selector][1:, 1:].astype(bool),
                logging.WARNING,  # if imageId != 36718140 else logging.INFO,
            )
            fl.solve(fcPairs=fcPairs, haveDummy=False)

            facilities = np.flatnonzero(fl.fData.data[:, FData.isOpen])
            openFacCosts = openCostSubset[facilities]
            cityConnectCosts = connectionCostSubset[1:, 1:][
                np.ix_(
                    facilities, np.isin(fl.cData.data[:, CData.facility], facilities)
                )
            ]

            # Cost for assigning TP label
            truePosConnectCosts = np.nansum(cityConnectCosts, axis=1) + openFacCosts

            # Cost for retaining original FP label (connecting cities to dummy)
            falsePosConnectCosts = (
                np.count_nonzero(np.nan_to_num(cityConnectCosts), axis=1)
                * connectionCostSubset[facilities + 1, 0]
            )

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
            self.annoStore.annotations.loc[
                :, BoxAggregator.normedBoxCoordCols
            ].to_numpy(),
        )
        # Find all boxes in the global set that do not intersect with a particular
        # ground truth box.
        # NOTE: Raises warning because bigBBoxDistances intentionally contains NaN
        # values (so comparison always returns false)
        self.bigBBoxMisses = (
            np.nan_to_num(self.bigBBoxDistances, 2)
            > self.initPhaseParameters["required_overlap_fraction"]
        )

        # The global set approximates the probability that any position in any
        # image contains a target object e.g. all likely targets may be
        # concentrated in the centre of the image.

        # Every global box that doesn't intersect with a ground truth box for
        # this image increases the probability that there is a real box missing
        # from the current ground truth estimate.

        # If the worker that provided the global box has a high false positive
        # probability, then this increases the chance that the missed
        # intersection implies a missed target.

        # If the cost of identifying the ground truth box as a target was high,
        # then this reduces the chance that the missed intersection implies a
        # missed target.

        # Recall that the cost is the sum of negated log false negative
        # probabilities over all workers who annotated the image corresponding
        # to a particluar box i.e. a higher cost implies many workers with a
        # low false negative probability annotated the image.
        self.bigBBoxMissContributions = (
            pd.DataFrame(self.bigBBoxMisses)
            .groupby(by=self.groundTruthBoxImageIds)
            .apply(
                lambda grp, fpp, oc: np.sum(
                    np.all(grp, axis=0) * (fpp) * np.exp(-oc.false_neg_prob)
                ),
                self.annoStore.annotations.false_pos_prob,
                self.openCosts,
            )
        )

        imageGroupedAnnotations = self.annoStore.annotations.groupby(by="image_id")

        # Compute expected false negatives based on annotations for particular
        # images.
        expNumFalseNegative = imageGroupedAnnotations.apply(
            self.computeImageExpNumFalseNegative
        )

        # Combine image-specific false negative count expectation with global
        # expectation.
        expNumFalseNegative.loc[
            self.bigBBoxMissContributions.index
        ] += self.bigBBoxMissContributions

        self.statStore.setImageExpNumFalseNegative(
            expNumFalseNegative, expNumFalseNegative.index
        )

    def computeExpNumFalsePositive(self):
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
        # to mark thr ground truth
        possibleWorkers = matchingAnnos.groupby(by="image_id").worker_id.apply(set)

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
        # Retrieve the false positive probabilities for workers that matched each ground truth
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

        # Retrieve the false negative probabilities for workers that matched each ground truth
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
        expNumFalsePositive = (
            combinedWorkers.reset_index()
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
        position exceeds $\pm \delta$ is given by $\mathrm{erfc}\left(\frac{\delta}{\sqrt{2\sigma^{2}}}\right)$.

        Note that the variances are expressed in 1-IoU coordinates and are therefore ounded in the range $[0,1]$
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
        summedCombinedVariances = combinedVariances.groupby(
            by=[
                self.annoStore.annotations.loc[
                    matchingAnnos.index, "image_id"
                ].to_numpy(),
                self.annoStore.annotations.loc[
                    matchingAnnos.index, "association"
                ].to_numpy(),
            ]
        ).sum()

        # Step 3: Compute the probability that each the positional error on each ground truth
        # exceeds a threshold overlap fraction.
        groundTruthInaccurateProbs = sp.special.erfc(
            (1 - self.lossParams["required_accuracy"])
            / np.sqrt(2 * summedCombinedVariances)
        )

        # Step 4: Compute the expected number of ground truth boxes in each image with positional
        # error that exceeds the threshold overlap fraction by summing the probabilities that each
        # ground truth box associated with the image does so.
        expectedNumInaccurate = groundTruthInaccurateProbs.groupby(level=0).sum()
        self.statStore.setImageExpNumInaccurate(
            expectedNumInaccurate, expectedNumInaccurate.index
        )

        return groundTruthInaccurateProbs.rename("inaccurate_prob")

    def processBatch(self):
        # ** Once-per-batch computations
        # Step 1: Compute 1-IOU distances between all annotations in normalised coordinates
        print("Computing IoU distances...")
        self.distances = self.imageProcessor.getIouDistances(
            store=self.annoStore, shuffle=False
        )
        # Step 2: Compute initial disallowed connection mask for all annotations
        print("Computing initial disallowed connection mask...")
        exclusiveIndexSetSizes = (
            self.annoStore.getMultiIndexGroupedAnnotations().count().iloc[:, 0].values
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
        print("Computing global box overlap statistics...")
        shuffledDistances = self.imageProcessor.getIouDistances(
            store=None, shuffle=True
        )
        overlaps = (
            shuffledDistances < self.initPhaseParameters["required_overlap_fraction"]
        )
        overlapCounts = overlaps.sum(axis=1) + 1
        falsePosProbs = (overlapCounts / overlapCounts.size).rename("false_pos_prob")
        self.bigBBoxSet = self.annoStore.annotations[
            ["x1_normed", "x2_normed", "y1_normed", "y2_normed"]
        ].merge(falsePosProbs, left_index=True, right_index=True)

        # Step 4:
        # ** Initial batch processing
        print("Initial batch processing...")
        self.processBatchStep(init=True)

        # Step 5:
        # ** Expectation maximisation
        currentLikelihood = -np.inf
        previousLikelihood = -np.inf
        for batchStep in range(10):
            print(f"Batch processing iteration {batchStep}...")
            self.statStore.resetAssociatedBoxes()
            self.statStore.resetGroundTruths()
            currentLikelihood = self.processBatchStep(init=False)
            if currentLikelihood <= previousLikelihood:
                print(f"Converged @ {currentLikelihood}")
                break
            else:
                print(
                    f"Not Converged @ {currentLikelihood} (Previous: {previousLikelihood})"
                )
                previousLikelihood = currentLikelihood

        # Step 6: Compute risks for all images.
        print("\tComputing risks...")
        # Step 6a: Compute expected number of false negatives for all images.
        print("\t\tComputing per-image expected false negative counts...")
        self.computeExpNumFalseNegative()

        # Step 6b: Compute expected number of false positives for all images.
        print("\t\tComputing per-image expected false positive counts...")
        gtFalsePosProbs = self.computeExpNumFalsePositive()

        # Step 6c: Compute expected variance of (assumed) true positive annotations.
        print("\t\tComputing per-image expected innacurate positive counts...")
        gtInaccurateProbs = self.computeExpNumInaccurate()

        # Step 6d: Compute image risks
        groundTruthRisks = self.lossParams["false_pos_loss"] * (
            gtFalsePosProbs.false_pos_prob
            + (
                gtInaccurateProbs
                * (
                    self.lossParams["false_neg_loss"]
                    + (1 - gtFalsePosProbs.false_pos_prob)
                    * self.lossParams["false_pos_loss"]
                )
            )
        ).rename("risk")

        # Step 6e: Save individual ground truth box statistics
        self.statStore.setGroundTruths(
            pd.concat(
                [gtFalsePosProbs, gtInaccurateProbs, groundTruthRisks], axis=1
            ).reset_index()
        )

        # Step 6f: Compute per image risks
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
