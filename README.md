# uNetCryo

Author: Maximilian Beckers

Deep learning for denoising of cryo-EM maps. The underlying network is a U-net architecture trained on several cryo-EM half maps. Code for the model is in model.py, for the training in main.py and for the predictions in prediction.py.

The network is implemented with Keras and Tensorflow. 

Cryo-EM maps are read with the mrcfile package: 
Burnley T, Palmer C & Winn M (2017) Recent developments in the CCP-EM software suite. Acta Cryst. D73:469–477. doi: 10.1107/S2059798317007859
