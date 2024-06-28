These results were obtained with model with the following modifications:
1. New loss function to predict eos tokens directly
2. Implementation of E_cut to discard clusters of energy below E < E_cut
3. Implementation of shuffling dataset between each epoch by parameter shuffle in config file