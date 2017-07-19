from models import Model, Poly2DModel
import numpy as np


class Ransac:
    def __init__(self, rank: int, points: list, outliers_ratio: float, m: Poly2DModel, separate=False):
        if separate is False:
            self.x_s, self.y_s = zip(points[0], points[1])
        else:
            self.x_s, self.y_s = points[0], points[1]
        self.outliers_ratio_ = outliers_ratio
        self.model_ = m
        self.sampleSize_ = rank  # how many points to sample at each iteration
        self.dataSize_ = len(self.x_s)
        self.inliers = [[], []]  # [xs:list,ys:list]
        self.tolerance = 0.5
        self.max_iterations = 1000
        self.iterating = False
        self.current_iteration = 0

    @property
    def tolerance(self):
        return self.tolerance_

    @tolerance.setter
    def tolerance(self, value):
        self.tolerance_ = value
        self.model_.tolerance = value

    def __iter__(self):
        return self

    def __next__(self):
        # sample points
        if self.iterating is False:
            self.iterating = True
            self.current_iteration = 1

        expected_inliers_ratio = 1 - self.outliers_ratio_
        inliers_ratio = len(self.inliers[0]) / self.dataSize_

        if inliers_ratio > expected_inliers_ratio or self.current_iteration > self.max_iterations:
            raise StopIteration

        indexes = np.random.randint(self.dataSize_, size=self.sampleSize_)
        x_s = [self.x_s[i] for i in indexes]
        y_s = [self.y_s[i] for i in indexes]

        # generate model
        self.model_.updateModel([x_s, y_s], separate=True)

        # inliers
        inliers_x, inliers_y = self.model_.filterOutliers(self.x_s, self.y_s)
        if len(inliers_x) > len(self.inliers[0]):  # got more inliers
            self.inliers = [inliers_x, inliers_y]

        self.current_iteration += 1

        # 1. return best_inliers
        # 2. sample_points for iteration
        # 3. inliers found in current iteration
        return self.inliers , [x_s, y_s], [inliers_x, inliers_y]

    def solve(self):
        expected_inliers_ratio = 1 - self.outliers_ratio_
        self.current_iteration = 1

        while True:
            inliers_ratio = len(self.inliers[0]) / self.dataSize_

            if inliers_ratio > expected_inliers_ratio or self.current_iteration > self.max_iterations:
                return self.inliers

            indexes = np.random.randint(self.dataSize_, size=self.sampleSize_)
            x_s = [self.x_s[i] for i in indexes]
            y_s = [self.y_s[i] for i in indexes]

            # generate model
            self.model_.updateModel([x_s, y_s], separate=True)

            # inliers
            inliers_x, inliers_y = self.model_.filterOutliers(self.x_s, self.y_s)
            if len(inliers_x) > len(self.inliers[0]):  # got more inliers
                self.inliers = [inliers_x, inliers_y]

            # next iteration
            self.current_iteration += 1


rn = Ransac(rank=2, points=[[1, 1.2, 1.1, 1.5], [1.1, 1.2, 2.4, 1.7]], outliers_ratio=0.3, m=Poly2DModel(),
            separate=True)
rn.max_iterations = 100
rn.tolerance = 0.5
solution = rn.solve()
z = np.polyfit(solution[0], solution[1], 1) # line
p = np.poly1d(z)
print(p)
