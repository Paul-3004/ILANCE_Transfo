# Particle-Flow reconstruction with Transformer 
In this project, we implement various architectures based on the original Transformer and use the models to reconstruct Monte Carlo particles from their associated hits in the detectors. In particular, we use the energy deposits and the positions of the hits to predict from which particles the hits were produced. The predictions include the charge, a unique integer number identifying the particles, their energies as well as directions. 

The project is built on Pytorch's API to build and train the models as well as the awkward library to handle ragged arrays. 

# Description of the branches
Several branches exist for this project and contain different network architectures and/or different training methods. 
- `version_1`  is the first model implemented and follows the transformer's original architecture
- `dev_MMHA` is a second implementation whith a slight modification to the architecture as described in [Network architectures](docs/NetworkArchitectures.md)
- `new_loss` reuses the same architecture as 'dev_MMHA' but implements a new loss function specialised to predict the next token's kind (`<bos>`, `<eos>`, `<sample>`, `<hits>`). This implies mandatory changes on the features of both hits and labels. The labels are also sorted by decreasing energy and a threshold is applied, discarding low energy clusters. 

# Documentation
- For any questions related to Pytorch's API or awkward library please refer to their corresponding documentation on [Pytorch's API](https://pytorch.org) and [Awkward Library](https://awkward-array.org/doc/main/)
- Some key methods/functions algorithm are explained in [Key Algorithms](docs/KeyAlgorithms.md)
- More information on the original transformer architecture as well as the variants implemented can be found in [Network Architectures](docs/NetworkArchitectures.md)
- More information on the datasets used as well as their preprocessing can be found in [Datasets and Preprocessing](docs/DatasetsPreprocessing.md)
- Results and performances of the models can be found in [Results](docs/Results.md)



