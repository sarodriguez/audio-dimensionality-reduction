# !pip install llvmlite==0.31.0
# !pip install llvmlite==0.34.0
# !pip install https://github.com/lmcinnes/umap/archive/0.5dev.zip
# !pip install pynndescent

from tensorflow.keras.datasets import mnist
from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
from umap.parametric_umap import umap_loss
from umap.umap_ import find_ab_params
import matplotlib.pyplot as plt
import tensorflow as tf


def train_param_umap(X, n_components, n_neighbors=10, distance_metric="euclidean", batch_size=250, epochs=200):
    """
    Train a Parametric UMAP model given the parameters. For more information visit  the repoository
    https://github.com/timsainb/ParametricUMAP_paper
    :param X:
    :param n_components:
    :param n_neighbors:
    :param distance_metric:
    :param batch_size:
    :param epochs:
    :return: Tuple (X transformed with the model, Model
    """
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric
    distance_metric = "euclidean"
    # number of neighbors for computing k-neighbor graph
    n_neighbors = 10

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=distance_metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # nearest neighbors and distances for each point in the dataset
    # np.shape(knn_indices), np.shape(knn_dists)


    # Fuzzy simplicial sets

    # get indices and distances
    # knn_indices, knn_dists = nnd.neighbor_graph
    random_state = check_random_state(None)
    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=distance_metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )


    # n_components = 2 # number of latent dimensions
    dims = X.shape[1:]
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=dims),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    encoder.summary()

    # Create batch iterator
    from umap.parametric_umap import construct_edge_dataset
    # n_epochs is used to compute epochs_per_sample, which, in non-parametric UMAP,
    # is the total number of epochs to optimize embeddings over. The computed value
    # epochs_per_sample, is the number of epochs  each edge is optimized over
    # (higher probability = more epochs).
    n_epochs = 200

    # batch_size = 1000 # iterate over batches of 1000 edges
    batch_size = 250 # iterate over batches of 1000 edges

    # get tensorflow dataset of edges
    (
        edge_dataset,
        batch_size,
        n_edges,
        head,
        tail,
        edge_weight,
    ) = construct_edge_dataset(
        X,
        umap_graph,
        n_epochs,
        batch_size,
        parametric_embedding = True,
        parametric_reconstruction = False,
    )

    (sample_edge_to_x, sample_edge_from_x), _ = next(iter(edge_dataset))

    min_dist = 0.1  # controls how tightly UMAP is allowed to pack points together (0 is more)
    _a, _b = find_ab_params(1.0, min_dist)
    negative_sample_rate = 5  # how many negative samples to train on per edge.

    umap_loss_fn = umap_loss(
        batch_size,
        negative_sample_rate,
        _a,
        _b,
        edge_weight,
        parametric_embedding=True
    )

    # define the inputs
    to_x = tf.keras.layers.Input(shape=dims, name="to_x")
    from_x = tf.keras.layers.Input(shape=dims, name="from_x")
    inputs = [to_x, from_x]

    # parametric embedding
    embedding_to = encoder(to_x)
    embedding_from = encoder(from_x)

    # concatenate to/from projections for loss computation
    embedding_to_from = tf.concat([embedding_to, embedding_from], axis=1)
    embedding_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(
        embedding_to_from
    )
    outputs = {'umap': embedding_to_from}

    # create model
    parametric_model = tf.keras.Model(inputs=inputs, outputs=outputs,)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    parametric_model.compile(
        optimizer=optimizer, loss=umap_loss_fn
    )


    # Fit model
    steps_per_epoch = int(
        n_edges / batch_size / 5
    )
    # create embedding
    z = encoder.predict(X)
    return z, encoder


# if __name__ == '__main__':
#     # # load dataset
#     (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#     X = train_images.reshape((60000, 28, 28, 1))
#     # X.shape
#     encoder = train_param_umap(X, n_components=2)