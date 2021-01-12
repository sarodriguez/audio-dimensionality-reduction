#########################################################
#     If you want to run with the synthetic dataset.    #
#########################################################
SYNTHETIC_DATASET_PATH="C:\Proy\VA\project\synthetic_dataset" # Path to the folder containing the wav audio samples
SYNTHETIC_FEATURES_PATH="C:\Proy\VA\project\data\synthetic_features" # Path to folder where the created features should be stored
SYNTHETIC_MODEL_OUTPUT_PATH="C:\Proy\VA\project\data\models" # Path to folder where the trained models and their results should be stored
python preprocess.py --spectrograms --raw --dataset=synthetic_dataset --dataset_path=$SYNTHETIC_DATASET_PATH --output_path=$SYNTHETIC_FEATURES_PATH
python dimensionality_reduction.py --input_type=raw_waveforms --output_path=$SYNTHETIC_MODEL_OUTPUT_PATH --preprocessed_dataset_path=$SYNTHETIC_FEATURES_PATH
python dimensionality_reduction.py --input_type=spectrograms --output_path=$SYNTHETIC_MODEL_OUTPUT_PATH --preprocessed_dataset_path=$SYNTHETIC_FEATURES_PATH

python dashboard/main.py --model_path=$SYNTHETIC_MODEL_OUTPUT_PATH --dataset_path=$SYNTHETIC_DATASET_PATH --preprocessed_dataset_path=$SYNTHETIC_FEATURES_PATH --dataset=synthetic_dataset
#########################################################
# If you want to run with the free spoken digit dataset #
#########################################################
FSDD_DATASET_PATH='C:\Proy\VA\project\data\free-spoken-digit-dataset\recordings' # Path to the folder containing the wav audio samples
FSDD_MODIFIED_DATASET_PATH='C:\Proy\VA\project\data\fsdd\data' # Path to the folder where the preprocessed audio samples should be stored
FSDD_FEATURES_PATH='C:\Proy\VA\project\data\fsdd\features' # Path to folder where the created features should be stored
FSDD_MODEL_OUTPUT_PATH='C:\Proy\VA\project\data\fsdd\models' # Path to folder where the trained models and their results should be stored
python preprocess.py --spectrograms --raw --dataset=free_spoken_digits_dataset --dataset_path=$FSDD_DATASET_PATH --output_path=$FSDD_FEATURES_PATH --mod_dataset_output_path=$FSDD_MODIFIED_DATASET_PATH
python dimensionality_reduction.py --input_type=raw_waveforms --output_path= --preprocessed_dataset_path=$FSDD_FEATURES_PATH
python dimensionality_reduction.py --input_type=spectrograms --output_path=$FSDD_MODEL_OUTPUT_PATH --preprocessed_dataset_path=$FSDD_FEATURES_PATH

python dashboard/main.py --model_path=$FSDD_MODEL_OUTPUT_PATH --dataset_path=$FSDD_MODIFIED_DATASET_PATH --preprocessed_dataset_path=$FSDD_FEATURES_PATH --dataset=free_spoken_digits_dataset


#########################################################
#            In order to run with embeddings            #
#########################################################
# This is a private dataset
EMBEDDING_PATH='C:\Proy\VA\project\data\embeddings\spk_embeddings.pickle' # Path to the embedding file
EMBEDDING_FEATURES_PATH='C:\Proy\VA\project\data\embeddings\features' # Path to folder where the created features should be stored
EMBEDDING_MODEL_OUTPUT_PATH='C:\Proy\VA\project\data\embeddings\models' # Path to folder where the trained models and their results should be stored
python preprocess_embeddings.py --embedding_path=$EMBEDDING_PATH --output_path=$EMBEDDING_FEATURES_PATH
python dimensionality_reduction.py --input_type=embeddings --output_path=$EMBEDDING_MODEL_OUTPUT_PATH --preprocessed_dataset_path=$EMBEDDING_FEATURES_PATH




