import numpy as np


class SOM:


    def __init__(self):
        pass


    def fit(self, X, y=None):
        pass


    @staticmethod
    def get_n_closest_idx(X, vector, n=1):

        distances = np.sqrt(np.sum((X - vector)**2, axis=-1))

        distances_shape = distances.shape
        distances = distances.flatten()
        
        n_closest_idx = []
        

        for i in range(n):

            idx = distances.argmin()
            distances[idx] = np.inf

            n_closest_idx.append(np.unravel_index(idx, distances_shape))


        return n_closest_idx




if __name__ == "__main__":

    som = SOM()


    a = np.array([[[1, 3, -1], [9, 0, 2]], [[8, 4, 3], [13, 4, -2]]])

    c = np.array([2, 8, 1])

    print(som.get_n_closest_idx(a, c, 2))






