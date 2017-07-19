"""General Template for Models to use RANSAC"""
import numpy as np
import warnings
import unittest

warnings.simplefilter('ignore', np.RankWarning)


class Model:
    def __init__(self, *args):
        self.points_, self.rank_, self.model_, self.coefficients_ = None, None, None, None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def isOutlier(self, x_s: list, y_s: list):
        raise NotImplementedError

    def filterOutliers(self, x_s: list, y_s: list):
        inliers = ~ self.isOutlier(x_s, y_s)
        return list(np.asarray(x_s)[inliers]), list(np.asarray(y_s)[inliers])

    def updateModel(self, *args):
        raise NotImplementedError

    @property
    def tolerance(self):
        return self.tolerance_

    @tolerance.setter
    def tolerance(self, value):
        self.tolerance_ = value


class Poly2DModel(Model):
    def __init__(self, points=None, separate=False):
        super().__init__()
        self.tolerance = 0.0
        if points is not None:
            self.updateModel(points, separate)

    def __call__(self, *args, **kwargs):
        if len(args) > 1:
            raise Exception('invalid number of arguments')
        if isinstance(args[0], float) or isinstance(args[0], int) or isinstance(args[0], list):
            return self.model_(args[0])
        else:
            raise TypeError

    def isOutlier(self, x_s, y_s):
        return abs(y_s - self.model_(x_s)) > self.tolerance

    def updateModel(self, points, separate=False):
        self.points_ = points
        self.rank_ = len(self.points_) - 1
        if separate is False:
            x, y = zip(points[0], points[1])
        else:
            x, y = points[0], points[1]
        self.coefficients_ = np.polyfit(x, y, self.rank_)
        self.model_ = np.poly1d(self.coefficients_)


class TestPolyModelsClass(unittest.TestCase):
    def setUp(self):
        self.p = Poly2DModel([[1, 1], [2, 2]])
        self.p.tolerance = 0.5

    def testGetValueForSingleValues(self):
        self.assertAlmostEqual(self.p(1.1), 1.1)

    def testGetValueForListValues(self):
        from_model = self.p([1.1, 2.5])
        expected = [1.1, 2.5]
        for i, value in enumerate(from_model):
            self.assertAlmostEqual(from_model[i], expected[i])

    def testOutlierDetection(self):
        x_s = [1.1, 1.4]
        y_s = [1.5, 2.9]
        self.assertListEqual(list(self.p.isOutlier(x_s, y_s)), [False, True])

    def testSelectInliers(self):
        x_s = [1.1, 1.4]
        y_s = [1.5, 2.9]
        self.assertListEqual(self.p.filterOutliers(x_s, y_s)[0], [1.1])  # xs
        self.assertListEqual(self.p.filterOutliers(x_s, y_s)[1], [1.5])  # ys


if __name__ == '__main__':
    unittest.main()
