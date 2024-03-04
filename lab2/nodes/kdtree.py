from scipy.spatial import cKDTree
import numpy as np

class KDTree:
    def __init__(self):
        self.points = []
        self.lazy = 0
        self.tree = None
    
    def size(self):
        return len(self.points)
    
    def insert(self, point):
        self.points.append(point)
        if self.size() - self.lazy > np.sqrt(self.size()):
            self.tree = cKDTree(self.points)
            self.lazy = self.size()
    
    def query(self, point):
        dmin, index = np.inf, None
        if self.tree is not None:
            dmin, index = self.tree.query(point)
        for i in range(self.lazy, self.size()):
            d = np.linalg.norm(point - self.points[i])
            if d < dmin:
                dmin, index = d, i
        return dmin, index
    
    def query_ball(self, point, radius):
        res = []
        if self.tree is not None:
            res = self.tree.query_ball_point(point, radius)
        for i in range(self.lazy, self.size()):
            d = np.linalg.norm(point - self.points[i])
            if d <= radius:
                res.append(i)
        return res
