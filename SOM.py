import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle

class SOM:

    class NeighbourhoodFunction:
        def __init__(self, function_name, learning_rate_start, learning_rate_end, epochs):
            self.learning_rate_start = learning_rate_start
            self.learning_rate_end = learning_rate_end
            self.epochs = epochs
            self.function_name = function_name

        def __call__(self, distances, time):
            scale = self.learning_rate_start*(self.learning_rate_end/self.learning_rate_start)**(time/self.epochs)

            tmp = np.power(distances / scale, 2)
            res = np.exp(-tmp/2)

            if self.function_name == "mexican_hat":
                return (1 - tmp) * res
            
            return res


    def __init__(self, nrows, ncols, 
                 epochs=10,
                 learning_rate_start=0.1,
                 learning_rate_end=0.01,
                 neighbourhood_rate_start=6,
                 neighbourhood_rate_end=2,
                 grid_type="hex",
                 neighbourhood_func="mexican_hat"):

        self.nrows = nrows
        self.ncols = ncols

        self.vectors = None
        self.categories = None

        self.epochs = epochs

        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end

        if not grid_type in ["hex", "rect"]:
            raise ValueError(f"Unknown grid type {grid_type}")
        self.grid_type = grid_type
        
        self.indexes = np.array([[[i, j] for j in range(ncols)] for i in range(nrows)])
        if self.grid_type == "hex":

            neighbourhood_rate_start += 1
            neighbourhood_rate_end += 1
            
            # axial axes for easy distances computation on hexagonal grid
            for i in range(self.indexes.shape[0]):
                if i % 2 == 0:
                    self.indexes[i, :, 1] *= 2
                else:
                    self.indexes[i, :, 1] += np.arange(1, ncols + 1)

        if not neighbourhood_func in ["mexican_hat", "gaussian"]:
            raise ValueError(f"Unknown neighbourhood function {neighbourhood_func}")
        self.neighbourhood_func = SOM.NeighbourhoodFunction(neighbourhood_func, 
                                                            neighbourhood_rate_start, 
                                                            neighbourhood_rate_end,
                                                            epochs)

    def fit(self, X, y=None):
        self.vectors = np.random.uniform(0, 1, (self.nrows, self.ncols, X.shape[1]))

        for t in range(self.epochs):
            learning_rate = self.learning_rate_start * (self.learning_rate_end/self.learning_rate_start)**(t/self.epochs)
            for i in range(X.shape[0]):
                bmu = self.get_n_closest_idx(self.vectors, X[i, :])[0]

                distances = self.get_grid_distances(bmu)
                self.vectors += learning_rate * np.expand_dims(self.neighbourhood_func(distances, t), 2) * (X[i, :] - self.vectors)
        
        if y is None:
            return
        
        data = np.expand_dims(X, 1)
        self.categories = np.empty((self.nrows, self.ncols), dtype=object)

        for i in range(self.vectors.shape[0]):
            for j in range(self.vectors.shape[1]):
                nearest_idx = self.get_n_closest_idx(data, self.vectors[i, j], n=10)
                nearest_idx = [idx[0] for idx in nearest_idx]
                nearest_categories = y[nearest_idx]
                categories, counts = np.unique(nearest_categories, return_counts=True)
                self.categories[i, j] = categories[counts.argmax()]


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
        if self.categories is None:
            raise Exception("This instance should be fitted with categories vector first!")

        unique_categories = np.unique(self.categories)
        color_dict = {unique_categories[i]: f"C{i}" for i in range(unique_categories.shape[0])}

        if self.grid_type == "hex":
            self.__plot_hex(color_dict)
        else:
            self.__plot_rect(color_dict)

        plt.axis("off")
        plt.title(title)
        if path:
            plt.savefig(path, dpi=600, bbox_inches="tight")
        else:
            plt.show()

    def __plot_hex(self, color_dict):

        radius = 1
        d = np.sqrt(3)*radius/2
        margin = 0.5
        width = 2*margin + d*(2*self.ncols + 1)
        height = 2*margin + (self.nrows + self.nrows//2 + self.nrows%2)*2*d/np.sqrt(3)

        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])

        for i in range(self.ncols):
            x = margin + d*(2*i + 1)
            y = height - margin - np.sqrt(3)*d/2
            for j in range(self.nrows):
                ax.add_patch(RegularPolygon((x, y), 
                                            6, 
                                            radius=radius, 
                                            lw=1, 
                                            edgecolor="black",
                                            facecolor=color_dict[self.categories[i, j]]))

                ax.annotate(str(self.categories[i, j]), 
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

    def __plot_rect(self, color_dict):

        d = 1
        margin = 1
        width = d*self.ncols + 2*margin
        height = d*self.nrows + 2*margin

        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])

        for i in range(self.ncols):
            x = margin + d*i
            y = margin + d
            for j in range(self.nrows):
                ax.add_patch(Rectangle((x, y), 
                                       width=d, 
                                       height=d, 
                                       lw=1, 
                                       edgecolor="black",
                                       facecolor=color_dict[self.categories[i, j]]))

                ax.annotate(str(self.categories[i, j]), 
                            (x + d/2, y + d/2), 
                            weight="bold", 
                            fontsize=10*d, 
                            ha="center", 
                            va="center")

                y += d




if __name__ == "__main__":

    som = SOM(5, 3, grid_type="hex")


    som.plot(title="Rect grid")






