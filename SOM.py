import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle

class SOM:

    """ 
    Self-Organizing Map for high dimensional data visualizations.

    Args:
        nrows, ncols            : dimensions of self-organizing map.
        epochs                  : number of epochs.
        learning_rate_start     : learning rate at first epoch, will be changed by rule lr_time = lr_start(lr_end/lr_start)^(time/epochs).
        learning_rate_end       : learning at last epoch.
        neighbourhood_rate_start: neighbourhood rate at first epoch, will be changed similarly as learning rate.
        neighbourhood_rate_end  : neighbourhood rate at last epoch.
        grid_type               : type of self-organizing map coordinate system, can be "hex" or "rect".
        neighbourhood_func      : type of neighbourhood function, can be "gaussian" or "mexican_hat".
        bootstrap               : if None, no bootstrap will be used, if positive float, then int(X.shape[0]*bootstrap) samples will be randomly drawn on each epoch.
    """

    class NeighbourhoodFunction:
        """ Help class representing neighbourhood function. """
        def __init__(self, function_name, neighbourhood_rate_start, neighbourhood_rate_end, epochs):
            self.neighbourhood_rate_start = neighbourhood_rate_start
            self.neighbourhood_rate_end = neighbourhood_rate_end
            self.epochs = epochs
            self.function_name = function_name

        def __call__(self, distances, time):
            scale = self.neighbourhood_rate_start*(self.neighbourhood_rate_end/\
                    self.neighbourhood_rate_start)**(time/self.epochs)

            x = np.power(distances/scale, 2)/2
            res = np.exp(-x)

            if self.function_name == "mexican_hat":
                return (1 - x) * res
            
            return res


    def __init__(self,
                 nrows, ncols, 
                 epochs=100,
                 learning_rate_start=1,
                 learning_rate_end=0.1,
                 neighbourhood_rate_start=3,
                 neighbourhood_rate_end=1,
                 grid_type="hex",
                 neighbourhood_func="gaussian",
                 bootstrap=None):

        self.nrows = nrows
        self.ncols = ncols

        self.vectors = None
        self.categories = None
        self.umatrix = None

        self.epochs = epochs

        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end

        if bootstrap and bootstrap <= 0:
            raise ValueError("bootstrap must be greater than 0")

        self.bootstrap = bootstrap

        if not grid_type in ["hex", "rect"]:
            raise ValueError(f"Unknown grid type {grid_type}")
        self.grid_type = grid_type
        
        self.indexes = np.array([[[i, j] for j in range(ncols)] for i in range(nrows)])
        if self.grid_type == "hex":
            # doubled coords for easier distances computation on hexagonal grid
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
        """ Fits map to data X, if category vector y provided each vector from map will get a category assigned. """
        self.vectors = np.random.uniform(np.min(X), np.max(X), (self.nrows, self.ncols, X.shape[1]))
        if self.bootstrap:
            n = int(X.shape[0] * self.bootstrap)
        else:
            n = None

        for t in range(self.epochs):
            if self.bootstrap:
                data = X[np.random.randint(0, X.shape[0], n), :]
            else:
                data = X
            learning_rate = self.learning_rate_start * (self.learning_rate_end/self.learning_rate_start)**(t/self.epochs)
            for i in range(data.shape[0]):
                bmu = self.__get_n_closest_idx(self.vectors, data[i, :])[0]

                distances = self.__get_grid_distances(bmu)
                self.vectors += learning_rate * np.expand_dims(self.neighbourhood_func(distances, t), 2) * (data[i, :] - self.vectors)
        
        self.__umatrix()
        
        if y is None:
            return
        
        data = np.expand_dims(X, 1)
        self.categories = np.empty((self.nrows, self.ncols), dtype=object)

        for i in range(self.vectors.shape[0]):
            for j in range(self.vectors.shape[1]):
                nearest_idx = self.__get_n_closest_idx(data, self.vectors[i, j], n=10)
                nearest_idx = [idx[0] for idx in nearest_idx]
                nearest_categories = y[nearest_idx]
                categories, counts = np.unique(nearest_categories, return_counts=True)
                self.categories[i, j] = str(categories[counts.argmax()])

    def __get_grid_distances(self, bmu_idx):
        """ Calculates distances from best matching unit(bmu) to other points on the grid and returns them as matrix. """
        if self.grid_type == "hex":
            # manhattan distance on hexagonal grid
            drow = np.abs(self.indexes[:, :, 0] - self.indexes[bmu_idx[0], bmu_idx[1], 0])
            dcol = np.abs(self.indexes[:, :, 1] - self.indexes[bmu_idx[0], bmu_idx[1], 1])
            distances = drow + np.maximum((dcol - drow)/2, 0)
        else:
            # manhattan distance on rectangular grid
            distances = np.sum(np.abs(self.indexes - bmu_idx), axis=2)
        
        return distances

    @staticmethod
    def __get_n_closest_idx(X, vector, n=1):
        """ Returns indexes of n closest vectors from X to vector. """

        distances = np.sqrt(np.sum((X - vector)**2, axis=2))

        distances_shape = distances.shape
        distances = distances.flatten()
        
        n_closest_idx = []
        

        for i in range(n):

            idx = distances.argmin()
            distances[idx] = np.inf

            n_closest_idx.append(np.unravel_index(idx, distances_shape))

        return np.array(n_closest_idx)

    def plot(self, title="", path=None, plot_umatrix=True, plot_categories=False, legend=False):
        """ 
        Plots umatrix learned from fit, if data categories were provided can also draw category plot. 
        
        Args:
            title          : main title of the plot.
            path           : if not None, generated plot will be saved to this path.
            plot_umatrix   : if True, umatrix will be shown.
            plot_categories: if True, categories plot will be shown, requires y argument in fit function.
            legend         : if True, categories plot will have a legend, otherwise cell will have category inside as text.
        
        """

        if not (plot_categories or plot_umatrix):
            raise Exception("At least one of plot_categories and plot_umatrix must True!")

        if self.umatrix is None:
            raise Exception("This instance should be fitted first!")

        if plot_categories and self.categories is None:
            raise Exception("This instance should be fitted with categories vector first!")

        if plot_categories and plot_umatrix:
            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            fig.suptitle(title)
            axs[0].set_title("Umatrix")
            axs[1].set_title("Categories")
        else:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]
            plt.title(title)

        if plot_umatrix:
            self.__plot_umatrix(fig, axs[0])

        if plot_categories:
            unique_categories = np.unique(self.categories)
            color_dict = {unique_categories[i]: f"C{i}" for i in range(unique_categories.shape[0])}

            i = int(plot_umatrix)
            if self.grid_type == "hex":
                self.__plot_hex(axs[i], color_dict, legend)
            else:
                self.__plot_rect(axs[i], color_dict, legend)

            axs[i].axis("off")
            if legend:
                handles, labels = axs[i].get_legend_handles_labels()
                d = dict(zip(labels, handles))
                handles, labels = np.array(list(d.values())), np.array(list(d.keys()))
                idx = np.argsort(labels)
                axs[i].legend(handles[idx], labels[idx], loc="center left", bbox_to_anchor=(1, 0.6))
                
        if path:
            plt.savefig(path, dpi=600, bbox_inches="tight")
        else:
            plt.show()

    def __plot_hex(self, ax, color_dict, legend):
        """ Plots hexagonal grid. """

        radius = 1
        d = np.sqrt(3)*radius/2
        margin = 0.5
        width = 2*margin + d*(2*self.ncols + 1)
        height = 2*margin + (self.nrows + self.nrows//2 + self.nrows%2)*2*d/np.sqrt(3)

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
                                            facecolor=color_dict[self.categories[j, i]],
                                            label=self.categories[j, i]))
                if not legend:
                    ax.annotate(str(self.categories[j, i]), 
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

    def __plot_rect(self, ax, color_dict, legend):
        """ Plots rectangular grid. """

        d = 1
        margin = 1
        width = d*self.ncols + 2*margin
        height = d*self.nrows + 2*margin

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
                                       facecolor=color_dict[self.categories[j, i]],
                                       label=self.categories[j, i]))
                if not legend:
                    ax.annotate(str(self.categories[j, i]), 
                                (x + d/2, y + d/2), 
                                weight="bold", 
                                fontsize=10*d, 
                                ha="center", 
                                va="center")

                y += d
    
    def __plot_umatrix(self, fig, ax):
        """ Plots umatrix of fitted map. """
        im = ax.imshow(self.umatrix, 
                       vmin=np.min(self.umatrix), 
                       vmax=np.max(self.umatrix), 
                       cmap=plt.get_cmap("Blues"))
        cbar = fig.colorbar(im, ax=ax, fraction=0.05 * self.umatrix.shape[0] / self.umatrix.shape[1])
        cbar.set_label("euclidean distance")
        ax.axis("off")
    
    def __umatrix(self):
        """ Calculates umatrix, a 2D array containing distances between high dimensional map vectors. """
        self.umatrix = np.zeros((2*self.nrows - 1, 2*self.ncols - 1))

        # maps coords from self.indexes to 2D coords from self.vectors
        # actually has sense only for hex grid, as it uses doubled coords in self.indexes
        # for rect grid it's just an identity operator
        coord_dict = {tuple(self.indexes[i, j]): [i, j] for i in range(self.nrows) for j in range(self.ncols)}

        if self.grid_type == "hex":
            neighbourhood = [[-1, -1], [-1, 1], [0, 2], [1, 1], [1, -1], [0, -2]]
        else:
            neighbourhood = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        hex_count = 0 # count variable, used for umatrix calculations for hex grid 
        for i in range(self.umatrix.shape[0]):
            for j in range(self.umatrix.shape[1]):

                if i % 2 == 0 and j % 2 == 0:

                    mapped_coords = self.indexes[i//2, j//2]
                    count = 0
                    distance = 0.0
                    for delta_coords in neighbourhood:
                        neighbour = coord_dict.get((mapped_coords[0] + delta_coords[0], 
                                                    mapped_coords[1] + delta_coords[1]), 
                                                    None)
                        if neighbour:
                            count += 1
                            distance += np.linalg.norm(self.vectors[i//2, j//2] -\
                                                       self.vectors[neighbour[0], neighbour[1]])
                    
                    self.umatrix[i, j] = distance/count # count will be > 0 by this moment

                elif i % 2 == 0 and j % 2 != 0:

                    vector1 = self.vectors[i//2, (j - 1)//2]
                    vector2 = self.vectors[i//2, (j + 1)//2]
                    self.umatrix[i, j] = np.linalg.norm(vector1 - vector2)

                elif i % 2 != 0 and j % 2 == 0:

                    vector1 = self.vectors[(i - 1)//2, j//2]
                    vector2 = self.vectors[(i + 1)//2, j//2]
                    self.umatrix[i, j] = np.linalg.norm(vector1 - vector2)

                else:
                    if self.grid_type == "hex":
                        if hex_count % 2 == 0:
                            self.umatrix[i, j] = np.linalg.norm(self.vectors[(i + 1)//2, (j - 1)//2] -\
                                                                self.vectors[(i - 1)//2, (j + 1)//2])
                        else:
                            self.umatrix[i, j] = np.linalg.norm(self.vectors[(i - 1)//2, (j - 1)//2] -\
                                                                self.vectors[(i + 1)//2, (j + 1)//2])
                        hex_count += 1

                    else:
                        distance1 = np.linalg.norm(self.vectors[(i - 1)//2, (j - 1)//2] - \
                                                   self.vectors[(i + 1)//2, (j + 1)//2])

                        distance2 = np.linalg.norm(self.vectors[(i - 1)//2, (j + 1)//2] - \
                                                   self.vectors[(i + 1)//2, (j - 1)//2])

                        self.umatrix[i, j] = (distance1 + distance2)/2



    
    def save(self, path):
        """ Saves all data of this instance to path. """
        attr_dict = {}
        for name, value in self.__dict__.items():
            if name == "indexes":
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                value = value.tolist()
            elif name == "neighbourhood_func":
                value = self.neighbourhood_func.function_name
            
            attr_dict[name] = value
        
        with open(path, "w") as file:
            json.dump(attr_dict, file)

    @staticmethod
    def load(path):
        """ Loads saved data and returns SOM intialized with it. """
        with open(path, "r") as file:
            attr_dict = json.load(file)
        
        special_attrs = {"vectors": None, "categories": None, "umatrix": None}
        for name in special_attrs:
            val = attr_dict.pop(name)
            special_attrs[name] = np.array(val) if val is not None else val
        
        som = SOM(**attr_dict)
        som.vectors = special_attrs["vectors"]
        som.categories = special_attrs["categories"]
        som.umatrix = special_attrs["umatrix"]

        return som



