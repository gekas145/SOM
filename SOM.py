import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

class SOM:


    def __init__(self, nrows, ncols, 
                 epochs=100,
                 grid_type="hex", 
                 neighbourhood_func="mexican_hat"):
        self.nrows = nrows
        self.ncols = ncols

        self.vectors = None

        self.epochs = epochs

        if not grid_type in ["hex", "rect"]:
            raise ValueError(f"Unknown grid type {grid_type}")
        self.grid_type = grid_type
        
        self.indexes = np.array([[[i, j] for j in range(ncols)] for i in range(nrows)])
        if self.grid_type == "hex":
            # axial axes for easy distances computation on hexagonal grid
            for i in range(self.indexes.shape[0]):
                if i % 2 == 0:
                    self.indexes[i, :, 1] *=2
                else:
                    self.indexes[i, :, 1] += np.arange(1, ncols + 1)

        if neighbourhood_func == "mexican_hat":
            self.neighbourhood_func = self.__mexican_hat 
        elif neighbourhood_func == "gaussian":
            self.neighbourhood_func = self.__gaussian
        else:
            raise ValueError(f"Unknown neighbourhood function {neighbourhood_func}")

        


    def fit(self, X, y=None):
        self.vectors = np.random.normal(0, 1, (nrows, ncols, X.shape[1]))
        for t in range(self.epochs):
            for i in range(X.shape[0]):
                bmu = self.get_n_closest_idx(self.vectors, X[i, :])

                distances = self.get_grid_distances(bmu)
                # check this
                # self.vectors += self.neighbourhood_func(distances) * (X[i, :] - self.vectors)


    def get_grid_distances(self, idx):
        if self.grid_type == "hex":
            # manhattan distance on hexagonal grid
            drow = np.abs(self.indexes[:, :, 0] - self.indexes[idx[0], idx[1], 0])
            dcol = np.abs(self.indexes[:, :, 1] - self.indexes[idx[0], idx[1], 1])
            distances = drow + np.maximum((dcol - drow)/2, 0)
        else:
            # manhattan distance on rectangular grid
            distances = np.sum(np.abs(self.indexes - idx), axis=2)
        
        return distances
    
    @staticmethod
    def __mexican_hat(x):
        pass

    @staticmethod
    def __gaussian(x):
        pass

    @staticmethod
    def get_n_closest_idx(X, vector, n=1):

        distances = np.sqrt(np.sum((X - vector)**2, axis=2))

        distances_shape = distances.shape
        distances = distances.flatten()
        
        n_closest_idx = []
        

        for i in range(n):

            idx = distances.argmin()
            distances[idx] = np.inf

            n_closest_idx.append(np.unravel_index(idx, distances_shape))


        return np.array(n_closest_idx)

    def plot(self, title="", path=None):
        # if self.vectors is None:
        #     raise Exception("This instance should be fitted first!")

        idx = np.array([2, 3])

        radius = 1
        d = np.sqrt(3)*radius/2
        margin = 0.5
        width = 2*margin + d*(2*self.ncols + 1)
        height = 2*margin + (self.nrows + self.nrows//2 + self.nrows%2)*2*d/np.sqrt(3)

        distances = self.get_grid_distances(idx)

        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])

        for i in range(self.ncols):
            x = margin + d*(2*i + 1)
            y = y = height - margin - np.sqrt(3)*d/2
            for j in range(self.nrows):
                ax.add_patch(RegularPolygon((x, y), 6, radius=radius, lw=1, edgecolor="black"))
                ax.annotate(str(round(distances[j, i])), 
                            (x, y), 
                            weight="bold", 
                            fontsize=10*d, 
                            ha="center", 
                            va="center")

                if j % 2 == 0:
                    x += d
                else:
                    x -= d
                y -= np.sqrt(3)*d

        plt.axis("off")
        plt.title(title)
        if path:
            plt.savefig(path, dpi=600, bbox_inches="tight")
        else:
            plt.show()




if __name__ == "__main__":

    som = SOM(8, 6)


    som.plot(title="Hex grid")






