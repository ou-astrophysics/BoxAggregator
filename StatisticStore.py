import pandas as pd
import numpy as np
from typing import Sequence

from SaveableStore import SaveableStore


class StatisticStore(SaveableStore):

    saveableAttrs = [
        "workerStatistics",
        "imageStatistics",
        "groundTruthStatistics",
        "associatedBoxStatistics",
    ]

    def __init__(self):
        self.workerStatistics = pd.DataFrame()
        self.imageStatistics = pd.DataFrame()

        # Remaining DatatFrames require more complicated instantiation
        self.resetGroundTruths()
        self.resetAssociatedBoxes()

        # A cache to preserve state between expectation maximisation iterations
        self.cache = dict()

    def cacheWorkers(self):
        self.cache.update(dict(workerStatistics=self.workerStatistics.copy(deep=True)))

    def cacheImages(self):
        self.cache.update(dict(imageStatistics=self.imageStatistics.copy(deep=True)))

    def restoreCachedWorkers(self):
        self.workerStatistics = self.cache.pop("workerStatistics")

    def restoreCachedImages(self):
        self.imageStatistics = self.cache.pop("imageStatistics")

    def getCachedWorkers(self):
        return self.cache.get("workerStatistics", None)

    def getCachedImages(self):
        return self.cache.get("imageStatistics", None)

    def clearCache(self, keys: Sequence = None):
        if keys is None:
            self.cache.clear()
        else:
            for key in keys:
                del self.cache[key]

    def getWorkerStatistics(self, cached=False):
        if cached:
            return self.cache["workerStatistics"]
        return self.workerStatistics

    def getImageStatistics(self, cached=False):
        if cached:
            return self.cache["imageStatistics"]
        return self.imageStatistics

    def addWorkers(self, annotationStore, priors, priorParams):
        newWorkerIds = pd.Index(
            annotationStore.getAnnotations().worker_id.unique()
        ).difference(self.workerStatistics.index)

        self.workerStatistics = pd.concat(
            [
                self.workerStatistics,
                pd.DataFrame(
                    dict(
                        # Static initial values
                        worker_id=newWorkerIds,
                        false_pos_prob_prior=priors["volunteer_skill"][
                            "false_pos_prob"
                        ],
                        false_neg_prob_prior=priors["volunteer_skill"][
                            "false_neg_prob"
                        ],
                        variance_prior=priors["volunteer_skill"]["variance"],
                        # Dynamic values updated on every iteration within a batch
                        false_pos_prob=priors["volunteer_skill"]["false_pos_prob"],
                        false_neg_prob=priors["volunteer_skill"]["false_neg_prob"],
                        variance=priors["volunteer_skill"]["variance"],
                        num_annos=0,
                        num_false_pos_trials=priorParams["volunteer_skill"][
                            "nBeta_false_pos"
                        ],
                        num_false_pos=priorParams["volunteer_skill"]["nBeta_false_pos"]
                        * priors["volunteer_skill"]["false_pos_prob"],
                        num_not_false_pos=priorParams["volunteer_skill"][
                            "nBeta_false_pos"
                        ]
                        * (1 - priors["volunteer_skill"]["false_pos_prob"]),
                        num_false_neg_trials=priorParams["volunteer_skill"][
                            "nBeta_false_neg"
                        ],
                        num_false_neg=priorParams["volunteer_skill"]["nBeta_false_neg"]
                        * priors["volunteer_skill"]["false_neg_prob"],
                        num_not_false_neg=priorParams["volunteer_skill"][
                            "nBeta_false_neg"
                        ]
                        * (1 - priors["volunteer_skill"]["false_neg_prob"]),
                        num_variance_trials=priorParams["volunteer_skill"][
                            "nInv_chisq_variance"
                        ],
                        variance_numerator=priorParams["volunteer_skill"][
                            "nInv_chisq_variance"
                        ]
                        * priors["volunteer_skill"]["variance"],
                    )
                ).set_index("worker_id"),
            ]
        )

    def addImages(self, annotationStore, priors, priorParams):
        newImageIds = pd.Index(
            annotationStore.getAnnotations().image_id.unique()
        ).difference(self.imageStatistics.index)

        self.imageStatistics = pd.concat(
            [
                self.imageStatistics,
                pd.DataFrame(
                    dict(
                        image_id=newImageIds,
                        variance_prior=priors["image_difficulty"]["variance"],
                        variance=priors["image_difficulty"][
                            "variance"
                        ],  # TODO: This is never set, but duplicated in anno store - would be more efficient here
                        box_variance=priors["image_difficulty"][
                            "variance"
                        ],  # Expected variance of a ground truth box
                        expected_num_false_neg=0,
                        expected_num_false_pos=0,
                        expected_num_inaccurate=0,
                        risk=np.infty,
                        open_cost=0,
                        num_variance_trials=priorParams["image_difficulty"][
                            "nInv_chisq_variance"
                        ],
                        variance_numerator=priorParams["image_difficulty"][
                            "nInv_chisq_variance"
                        ]
                        * priors["image_difficulty"]["variance"],
                        is_finished=False,
                    )
                ).set_index("image_id"),
            ]
        )

    def addAnnotations(self, annotationStore, priors, priorParams):
        self.addWorkers(annotationStore, priors, priorParams)
        self.addImages(annotationStore, priors, priorParams)

    def setWorkerSkills(self, skills, workers=slice(None)):
        self.workerStatistics.loc[
            workers, ["false_pos_prob", "false_neg_prob", "variance"]
        ] = skills

    def setImageVariances(self, variances, images=slice(None)):
        self.imageStatistics.loc[images, ["variance"]] = variances

    def setImageBoxVariances(self, variances, images=slice(None)):
        self.imageStatistics.loc[images, ["box_variance"]] = variances

    def setImageExpNumFalseNegative(self, expNumFalseNegatives, images=slice(None)):
        self.imageStatistics.loc[
            images, ["expected_num_false_neg"]
        ] = expNumFalseNegatives

    def setImageExpNumFalsePositive(self, expNumFalsePositives, images=slice(None)):
        self.imageStatistics.loc[
            images, ["expected_num_false_pos"]
        ] = expNumFalsePositives

    def setImageExpNumInaccurate(self, expNumInaccurate, images=slice(None)):
        self.imageStatistics.loc[images, ["expected_num_inaccurate"]] = expNumInaccurate

    def setImageRisk(self, imageRisk, images=slice(None)):
        self.imageStatistics.loc[images, ["risk"]] = imageRisk

    def setImageOpenCosts(self, openCost, images=slice(None)):
        self.imageStatistics.loc[images, "open_cost"] = openCost

    # Erases all associated box data.
    def resetAssociatedBoxes(self):
        self.associatedBoxStatistics = pd.DataFrame(
            columns=[
                "image_id",
                "annotation_id",
                "false_pos_prob",  # same as worker
                "false_neg_prob",  # same as worker
                "image_variance",  # variance based on image stats
                "combined_variance",  # variance based on combination of worker and image stats
                "variance_weighting",  # The weighting for the combination
            ]
        ).set_index(["image_id", "annotation_id"])

    # Erases all ground truth data.
    def resetGroundTruths(self):
        self.groundTruthStatistics = pd.DataFrame(
            columns=[
                "image_id",
                "association",
                "false_pos_prob",  # probability that gt is a false pos
                "inaccurate_prob",  # probability that the gt is inaccurate
                "risk",
            ]
        ).set_index(["image_id", "association"])

    # Intended for use with multiple batches when finished image ground truths
    # should not be erased. Probably not correct.
    def setAssociatedBoxes(self, associatedBoxAnnotations):
        newImageIds = pd.Index(associatedBoxAnnotations.image_id.unique()).difference(
            self.associatedBoxStatistics.index.get_level_values(level=0)
        )

        # find and remove any images for which the ground truth is being replaced.
        if self.associatedBoxStatistics.size > 0:
            replacements = pd.Index(
                associatedBoxAnnotations.image_id.unique()
            ).intersection(self.associatedBoxStatistics.index.get_level_values(level=0))

            if replacements.size > 0:
                self.associatedBoxStatistics.drop(
                    index=replacements, level=0, inplace=True
                )

        self.associatedBoxStatistics = (
            pd.concat(
                [
                    self.associatedBoxStatistics.reset_index(),
                    associatedBoxAnnotations.loc[
                        :,
                        [
                            "image_id",
                            "annotation_id",
                            "false_pos_prob",  # same as worker
                            "false_neg_prob",  # same as worker
                            "image_variance",  # variance based on image stats
                            "combined_variance",  # variance based on combination of worker and image stats
                            "variance_weighting",  # The weighting for the combination
                        ],
                    ],
                ]
            )
            .astype({"image_id": np.int64, "annotation_id": np.int64})
            .set_index(["image_id", "annotation_id"])
        )

    # Intended for use with multiple batches when finished image ground truths
    # should not be erased. Probably not correct.
    def setGroundTruths(self, groundTruthData):
        newImageIds = pd.Index(groundTruthData.image_id.unique()).difference(
            self.groundTruthStatistics.index.get_level_values(level=0)
        )

        # find and remove any images for which the ground truth is being replaced.
        if self.groundTruthStatistics.size > 0:
            replacements = pd.Index(groundTruthData.image_id.unique()).intersection(
                self.groundTruthStatistics.index.get_level_values(level=0)
            )

            if replacements.size > 0:
                self.groundTruthStatistics.drop(
                    index=replacements, level=0, inplace=True
                )

        self.groundTruthStatistics = (
            pd.concat(
                [
                    self.groundTruthStatistics.reset_index(),
                    groundTruthData.loc[
                        :,
                        [
                            "image_id",
                            "association",
                            "false_pos_prob",  # probability that gt is a false pos
                            "inaccurate_prob",  # probability that the gt is inaccurate
                            "risk",
                        ],
                    ],
                ]
            )
            .astype({"image_id": np.int64, "association": np.int64})
            .set_index(["image_id", "association"])
        )
