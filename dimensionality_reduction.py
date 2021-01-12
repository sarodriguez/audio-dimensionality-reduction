import os
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap
import argparse
from utilities import dir_path
import pickle



def similarity(args):
    input_type = args.input_type
    dataset_path = args.preprocessed_dataset_path
    output_path = args.output_path
    file_name = input_type + '.npy'

    dataset = np.load(Path(dataset_path, file_name))
    from sklearn.metrics.pairwise import euclidean_distances
    import pandas as pd
    import matplotlib.pyplot as plt
    dataset_raw = np.load(Path(dataset_path, 'raw_waveforms.npy'))
    if len(dataset.shape) == 3:
        dataset = dataset.reshape(dataset.shape[0], -1)
    metadata = pd.read_csv(Path(dataset_path, 'metadata.csv'))
    metadata = metadata.sort_values(by=['5', '6', '1'], ascending=[True, True, True])
    # metadata = metadata[~(metadata['6'] == 'combination')]
    ed = euclidean_distances(dataset[metadata.index], dataset[metadata.index])
    ed2 = euclidean_distances(dataset_raw[metadata.index], dataset_raw[metadata.index])
    sample_length = len(metadata)
    unique_wavetypes = len(metadata['5'].unique())
    ticks = [i*(sample_length/unique_wavetypes) + (sample_length/(unique_wavetypes*2)) for i in range(unique_wavetypes)]
    tick_labels = list(metadata['5'].unique())
    # Sort by wave


    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Euclidean Distances')
    ax[0].imshow(ed)
    ax[0].set_title('Spectrograms')
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(tick_labels)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(tick_labels)
    ax[1].imshow(ed2)
    ax[1].set_title('Raw Waveforms')
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(tick_labels)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(tick_labels)
    fig.show()
    # plt.imshow(ed)
    # plt.title('Euclidean Distances: Spectrograms')
    # plt.xticks(ticks, tick_labels)
    # plt.yticks(ticks, tick_labels)
    # plt.show()
    # plt.imshow(ed2)
    # plt.title('Euclidean Distances: Raw Waveforms')
    # plt.xticks(ticks, tick_labels)
    # plt.yticks(ticks, tick_labels)
    # plt.show()




def run_dimensionality_reduction(args):
    """
    Function that executes the default version of multiple dimensionality reduction algorithms over previously
    preprocessed datasets. The trained models and their results are stored in the specified output_path.
    So far we are using PCA, TruncatedSVD, UMAP and Parametric UMAP
    :param args:
    :return:
    """
    input_type = args.input_type
    dataset_path = args.preprocessed_dataset_path
    output_path = args.output_path
    file_name = input_type + '.npy'

    dataset = np.load(Path(dataset_path, file_name))
    # Flatten spectrograms in the input type is spectrograms
    if input_type == 'spectrograms':
        dataset = dataset.reshape(dataset.shape[0], -1)
    results_path = Path(output_path, input_type, 'results')
    models_path = Path(output_path, input_type, 'models')
    if not os.path.exists(Path(output_path, input_type)):
        os.makedirs(Path(output_path, input_type))
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pca = PCA(2)
    pca_results = pca.fit_transform(dataset)
    np.save(Path(results_path, 'pca.npy'), pca_results)

    # tsne = TSNE(2)
    # tsne_results = tsne.fit_transform(dataset)
    # np.save(Path(output_path, input_type, 'results', 'tsne.npy'), tsne_results)
    # with open(Path(models_path, 'tsne.pickle'), 'wb') as handle:
    #     pickle.dump(tsne, handle, protocol=pickle.HIGHEST_PROTOCOL)

    umapalg = umap.UMAP(n_components=2)
    umap_results = umapalg.fit_transform(dataset)
    np.save(Path(output_path, input_type, 'results', 'umap.npy'), umap_results)

    tsvd = TruncatedSVD(2)
    tsvd_results = tsvd.fit_transform(dataset)
    np.save(Path(results_path, 'tsvd.npy'), tsvd_results)

    # We were not able to execute param UMAP run in our development environment, and thus, we did it using a Google colab Notebook
    # https://colab.research.google.com/drive/1L0W9IibHMPw3k35k1NSqz4HuN-ffKUgM
    # from models.parametric_umap import train_param_umap
    # param_umap_results, param_umap = train_param_umap(dataset, n_components=2)

    if input_type != 'embeddings':
        with open(Path(models_path, 'pca.pickle'), 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(Path(models_path, 'umap.pickle'), 'wb') as handle:
            pickle.dump(umapalg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(Path(models_path, 'tsvd.pickle'), 'wb') as handle:
            pickle.dump(tsvd, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute and save dimensionality reduction 2D results.')
    parser.add_argument('--input_type', choices=['spectrograms', 'raw_waveforms', 'embeddings'],
                        help='Determines the type of input to be used for the dimensionality reduction techniques. '
                             'Options: [spectrograms, raw_waveforms, embeddings]')
    parser.add_argument('--batches', action='store_true',
                        help='Determines if batches are required for the output files'
                             '. This determines the output type. Numpy (not batches) '
                             'PyTorch dataset(batches)')
    parser.add_argument('--output_path', type=dir_path,
                        help='Specify the absolute output path to store the results for the dim. reduction techniques.')
    parser.add_argument('--preprocessed_dataset_path', type=dir_path,
                        help='Specify the absolute path (folder) containing the preprocessed audio data.')
    # parser.add_argument('--dataset', choices=['free_spoken_digits_dataset', 'synthetic_dataset'],
    #                     default='free_spoken_digits_dataset',
    #                     help='Specify one of the supported datasets [free_spoken_digits_dataset]')
    args = parser.parse_args()
    run_dimensionality_reduction(args)

