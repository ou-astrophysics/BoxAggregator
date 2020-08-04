import numba
import numpy as np
import pandas as pd

@numba.jit(nopython=True)
def computeIouDistancesImpl(normedBoxCoords):

    distances = np.empty(
        (normedBoxCoords.shape[0], normedBoxCoords.shape[0]), dtype=np.float32
    )

    for left in range(distances.shape[0]):
        x1l, x2l, y1l, y2l = normedBoxCoords[left]
        distances[left, left] = np.nan
        for right in range(left + 1, distances.shape[1]):
            x1r, x2r, y1r, y2r = normedBoxCoords[right]
            # Maximised @ 1 when IoU = 0, i.e. no overlap
            distances[left, right] = distances[right, left] = 1.0 - (
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


class ImageProcessor:
    def __init__(self):
        self.distances = None

    def computeIouDistances(self, image=None, store=None):
        if image is not None:
            normedBoxCoords = image.getAnnotations()[
                ["x1_normed", "x2_normed", "y1_normed", "y2_normed"]
            ]
        elif store is not None:
            normedBoxCoords = store.getAnnotations()[
                ["x1_normed", "x2_normed", "y1_normed", "y2_normed"]
            ]

        distances = pd.DataFrame(
            computeIouDistancesImpl(normedBoxCoords.values.astype(np.float32)),
            index=normedBoxCoords.index,
            columns=normedBoxCoords.index,
        )
        return distances

    def getIouDistances(self, image=None, store=None, shuffle=False):
        distances = self.distances
        # passing a datastore triggers recomputation
        if store is not None:
            distances = self.distances = self.computeIouDistances(
                image=image, store=store
            )
        elif self.distances is not None and image is not None:
            # can extract subset of existing distances
            distances = self.distances.loc[
                image.getAnnotationIndices(), image.getAnnotationIndices()
            ]

        if shuffle:
            permutedIndex = np.random.permutation(distances.index)
            return distances.loc[permutedIndex, permutedIndex], permutedIndex
        return distances
