# Single Cortical Neurons as Deep Networks  
This repo contains the code behind the work "[Single Cortical Neurons as Deep Artificial Neural Networks](https://www.cell.com/neuron/fulltext/S0896-6273(21)00501-8)"  

![Graphical Abstract](https://i.im.ge/2021/08/13/mybwx.png)

## Single Cortical Neurons as Deep Artificial Neural Networks  
David Beniaguev, Idan Segev, Michael London

**Abstract**: *Utilizing recent advances in machine learning, we introduce a systematic approach to characterize neurons’ input/output (I/O) mapping complexity. Deep neural networks (DNNs) were trained to faithfully replicate the I/O function of various biophysical models of cortical neurons at millisecond (spiking) resolution. A temporally convolutional DNN with five to eight layers was required to capture the I/O mapping of a realistic model of a layer 5 cortical pyramidal cell (L5PC). This DNN generalized well when presented with inputs widely outside the training distribution. When NMDA receptors were removed, a much simpler network (fully connected neural network with one hidden layer) was sufficient to fit the model. Analysis of the DNNs’ weight matrices revealed that synaptic integration in dendritic branches could be conceptualized as pattern matching from a set of spatiotemporal templates. This study provides a unified characterization of the computational complexity of single neurons and suggests that cortical networks therefore have a unique architecture, potentially supporting their computational power.*

## Resources
Neuron version of paper: [https://www.cell.com/neuron/fulltext/S0896-6273(21)00501-8](https://www.cell.com/neuron/fulltext/S0896-6273(21)00501-8)  
Open Access (slightly older) version of Paper: [https://www.biorxiv.org/content/10.1101/613141v2](https://www.biorxiv.org/content/10.1101/613141v2)  
Dataset and pretrained networks: [https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data)  
Dataset for training new models: [https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-train-data](https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-train-data)  
Notebook with main result: [https://www.kaggle.com/selfishgene/single-neuron-as-deep-net-replicating-key-result](https://www.kaggle.com/selfishgene/single-neuron-as-deep-net-replicating-key-result)  
Notebook exploring the dataset: [https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron](https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron)  
Twitter thread for short visual summery #1: [https://twitter.com/DavidBeniaguev/status/1131890349578829825](https://twitter.com/DavidBeniaguev/status/1131890349578829825)  
Twitter thread for short visual summery #2: [https://twitter.com/DavidBeniaguev/status/1426172692479287299](https://twitter.com/DavidBeniaguev/status/1426172692479287299)  
Figure360, author presentation of Figure 2 from the paper: [https://www.youtube.com/watch?v=n2xaUjdX03g](https://www.youtube.com/watch?v=n2xaUjdX03g)  
  

![single neuron as deep net illustration](https://user-images.githubusercontent.com/11506338/77857795-71a27b00-7208-11ea-937a-74cb6e414281.PNG)  

## Integrate and Fire code
- Use `integrate_and_fire_figure_replication.py` to simulate, fit, evaluate and replicate the introductory figure in the paper (Fig. 1)

## Single cortical neuron simulation code
![single neuron simulation illustration image](https://pbs.twimg.com/media/D7U15SSXoAAM-Js?format=jpg&name=4096x4096)
- Use `simulate_L5PC_and_create_dataset.py` to simulate a single neuron
  - All major parameters are documented inside the file using comments  
  - All necessary NEURON `.hoc` and `.mod` simulation files are under the folder `L5PC_NEURON_simulation\`
- Use `dataset_exploration.py` to explore the generated dataset
- Alternativley, just download the [data](https://www.kaggle.com/selfishgene/single-neurons-as-deep-nets-nmda-test-data) from kaggle, and look at exploration [script](https://www.kaggle.com/selfishgene/exploring-a-single-cortical-neuron)

### Note: we use the NEURON simulator for the L5PC simulation. More details about NEURON below
- neuron github repo: [https://github.com/neuronsimulator/nrn](https://github.com/neuronsimulator/nrn)  
- recommended introductory NEURON tutorial: [https://github.com/orena1/NEURON_tutorial
](https://github.com/orena1/NEURON_tutorial)  
- official NEURON with python tutorial: [https://neuron.yale.edu/neuron/static/docs/neuronpython/index.html](https://neuron.yale.edu/neuron/static/docs/neuronpython/index.html)  
- NEURON help fortum: [https://www.neuron.yale.edu/phpBB/index.php?sid=31f0839c5c63ca79d80790460542bbf3](https://www.neuron.yale.edu/phpBB/index.php?sid=31f0839c5c63ca79d80790460542bbf3)  

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

If you use this code or dataset, please cite the following two works:  

1. David Beniaguev, Idan Segev and Michael London. 2019. “Single Cortical Neurons as Deep Artificial Neural Networks.” BioRxiv, 613141. doi:10.1101/613141.
1. Hay, Etay, Sean Hill, Felix Schürmann, Henry Markram, and Idan Segev. 2011. “Models of Neocortical
Layer 5b Pyramidal Cells Capturing a Wide Range of Dendritic and Perisomatic Active Properties.”
Edited by Lyle J. Graham. PLoS Computational Biology 7 (7): e1002107.
doi:10.1371/journal.pcbi.1002107.
