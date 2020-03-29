# Single Cortical Neurons as Deep Networks  
This repo contains the code behind the work "[Single Cortical Neurons as Deep Artificial Neural Networks](https://www.biorxiv.org/content/10.1101/613141v2)"  

![single neuron as deep net illustration](https://user-images.githubusercontent.com/11506338/77857795-71a27b00-7208-11ea-937a-74cb6e414281.PNG)

**Single Cortical Neurons as Deep Artificial Neural Networks**  
David Beniaguev, Idan Segev, Michael London

Abstract: *We introduce a novel approach to study neurons as sophisticated I/O information processing units by utilizing recent advances in the field of machine learning. We trained deep neural networks (DNNs) to mimic the I/O behavior of a detailed nonlinear model of a layer 5 cortical pyramidal cell, receiving rich spatio-temporal patterns of input synapse activations. A Temporally Convolutional DNN (TCN) with seven layers was required to accurately, and very efficiently, capture the I/O of this neuron at the millisecond resolution. This complexity primarily arises from local NMDA-based nonlinear dendritic conductances. The weight matrices of the DNN provide new insights into the I/O function of cortical pyramidal neurons, and the approach presented can provide a systematic characterization of the functional complexity of different neuron types. Our results demonstrate that cortical neurons can be conceptualized as multi-layered “deep” processing units, implying that the cortical networks they form have a non-classical architecture and are potentially more computationally powerful than previously assumed.*

## Resources
Paper: [https://www.biorxiv.org/content/10.1101/613141v2](https://www.biorxiv.org/content/10.1101/613141v2)  
Dataset and pretrained networks: [https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data)  
Notebook with main result: [https://www.kaggle.com/selfishgene/single-neuron-as-deep-net-replicating-key-result](https://www.kaggle.com/selfishgene/single-neuron-as-deep-net-replicating-key-result)  
Notebook exploring the dataset: [https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron](https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron)  
Twitter thread for short visual summery: [https://twitter.com/DavidBeniaguev/status/1131890349578829825](https://twitter.com/DavidBeniaguev/status/1131890349578829825)  

## Single neuron simulation code
![single neuron simulation illustration image](https://pbs.twimg.com/media/D7U15SSXoAAM-Js?format=jpg&name=4096x4096)
- Use `simulate_L5PC_and_create_dataset.py` to simulate a single neuron
  - All major parameters are documented inside the file using comments  
  - All necessary NEURON `.hoc` and `.mod` simulation files are under the folder `L5PC_NEURON_simulation\`
- Use `dataset_exploration.py` to explore the generated dataset
- Alternativley, just download the [data](https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data) from kaggle, and look at exploration [script](https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron)

## TCN fitting and evaluation code
![fitting TCN to single neurons](https://pbs.twimg.com/media/D7U4O3HWsAI2YKK?format=png&name=900x900)
- `fit_CNN.py` contains the code used to fit a network to the dataset  
- `evaluate_CNN_test.py` and `evaluate_CNN_valid.py` contains the code used to evaluate the performace of the networks on test and validation sets

## Analysis code
- Use `main_figure_replication.py` to replicate the main figures (Fig. 2 & Fig. 3) after generating data and training models
- `valid_fitting_results_exploration.py` can be used to explore the fitting results on validation dataset
- `test_accuracy_vs_complexity.py` can be used to generate the model accuracy vs complexity plots (Fig. S5)
- Alternativley, visit key figure replication [notebook](https://www.kaggle.com/selfishgene/single-neuron-as-deep-net-replicating-key-result) on kaggle


## Acknowledgements
We thank Oren Amsalem, Guy Eyal, Michael Doron, Toviah Moldwin, Yair Deitcher, Eyal Gal and all lab members of the Segev and London Labs for many fruitful discussions and valuable feedback regarding this work.

If you use this dataset, please cite the following two works:  

1. David Beniaguev, Idan Segev and Michael London. 2019. “Single Cortical Neurons as Deep Artificial Neural Networks.” BioRxiv, 613141. doi:10.1101/613141.
1. Hay, Etay, Sean Hill, Felix Schürmann, Henry Markram, and Idan Segev. 2011. “Models of Neocortical
Layer 5b Pyramidal Cells Capturing a Wide Range of Dendritic and Perisomatic Active Properties.”
Edited by Lyle J. Graham. PLoS Computational Biology 7 (7): e1002107.
doi:10.1371/journal.pcbi.1002107.
