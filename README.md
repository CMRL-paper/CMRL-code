This repository contains the code and the dataset for our  paper.

> The paper will be published later.

### Dataset

For the contrastive learning task between programming languages and assembly instructions, this paper has constructed a dataset, which can be accessed at the following link: [CMRL Dataset on Hugging Face](https://huggingface.co/datasets/CMRL-paper/CMRL-dataset).

### Artifacts

Description of folders:

- IdaPythonScript：The  folder contains scripts for processing binary ELF files. These scripts provide code implementations to extract Control Flow Graphs (CFG) and assembly instructions from ELF files.
- model：The  contains the model design and method implementation code. Specifically, the `model.py` file includes the implementation of the APECL-Asm tokenizer, as well as the APECL-Asm encoder and the neural network model for the semantic structure-aware network.
- dataloader: The  folder contains code for dataset preprocessing and the implementation of the dataloader used during model training.
- modelTrain: The  folder contains the code for training models. This includes the APECL (Assembly Code-Programming Language Coordinated Representations Learning) method and the GBFEM (Graph Neural Network-Based Binary Function Embedding Generation Method).
  - `ApeclTrain.py`: This file contains the training code for the APECL method, which facilitates coordinated representation learning between assembly code and programming languages for binary code.
  - `GBFEMTrain.py`: This file contains the training code for the GBFEM method, which uses a graph neural network to generate embedding vectors for binary functions.
- exp: The  folder contains the code implementation for the similarity detection evaluation experiments.

### Feedback

If you need help or find any bugs, feel free to submit GitHub issues or PRs.