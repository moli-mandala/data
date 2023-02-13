# models

This folder contains code for training models on reflex prediction task on the Jambu dataset. Currently, since reconstructions (or Sanskrit etyma) are only available for all Indo-Aryan cognate sets, reflex prediction can only be done for that family.

Overview of the pipeline:

1. `preprocess.py`: Loads CLDF database into the necessary format for model training, i.e. filters and tokenises data, converts the phonemic symbols into numbers using a mapping, and saves the resulting dataset as a pickle.
2. `model.py`: Contains definitions for the following model architectures:
    - A GRU encoder-decoder with attention (implemented following [this tutorial](https://jasmijn.ninja/annotated_encoder_decoder/)).
    - TODO: A transformer.
3. `train.py`: Loads a pickled dataset and a model, and trains it. Performance and hyperparameters are recorded on Weights and Biases.