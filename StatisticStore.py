import pandas as pd
import numpy as np


class StatisticStore:
    def __init__(self):
        self.workerStatistics = pd.DataFrame()
        self.imageStatistics = pd.DataFrame()
        self.groundTruthStatistics = pd.DataFrame(
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

    def addWorkers(self, annotationStore, priors):
        newWorkerIds = pd.Index(
            annotationStore.getAnnotations().worker_id.unique()
        ).difference(self.workerStatistics.index)

        self.workerStatistics = pd.concat(
            [
                self.workerStatistics,
                pd.DataFrame(
                    dict(
                        worker_id=newWorkerIds,
                        false_pos_prob_prior=priors["volunteer_skill"][
                            "false_pos_prob"
                        ],
                        false_neg_prob_prior=priors["volunteer_skill"][
                            "false_neg_prob"
                        ],
                        variance_prior=priors["volunteer_skill"]["variance"],
                        false_pos_prob=priors["volunteer_skill"]["false_pos_prob"],
                        false_neg_prob=priors["volunteer_skill"]["false_neg_prob"],
                        variance=priors["volunteer_skill"]["variance"],
                    )
                ).set_index("worker_id"),
            ]
        )

    def addImages(self, annotationStore, priors):
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
                        variance=priors["image_difficulty"]["variance"], # TODO: This is never set, but duplicated in anno store - would be more efficient here
                        box_variance=priors["image_difficulty"]["variance"], # Expected variance of a ground truth box
                        expected_num_false_neg=0,
                    )
                ).set_index("image_id"),
            ]
        )

    def addAnnotations(self, annotationStore, priors):
        self.addWorkers(annotationStore, priors)
        self.addImages(annotationStore, priors)

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

    def setGroundTruths(self, groundTruthAnnotations):
        newImageIds = pd.Index(groundTruthAnnotations.image_id.unique()).difference(
            self.groundTruthStatistics.index.get_level_values(level=0)
        )

        # find and remove any images for which the ground truth is being replaced.
        if self.groundTruthStatistics.size > 0:
            replacements = pd.Index(
                groundTruthAnnotations.image_id.unique()
            ).intersection(self.groundTruthStatistics.index.get_level_values(level=0))

            if replacements.size > 0:
                self.groundTruthStatistics.drop(
                    index=replacements, level=0, inplace=True
                )

        self.groundTruthStatistics = (
            pd.concat(
                [
                    self.groundTruthStatistics.reset_index(),
                    groundTruthAnnotations.loc[
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
