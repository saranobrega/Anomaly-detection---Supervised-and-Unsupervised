# Anomaly detection Project
Created a new unsupervised method to detect anomaly samples using Isolation Forest and ANN

## Goal
![alt text](goal.png)
The goal here is to know what type of anomalies the models are spotting through the 2 different methods.
There are two different anomaly model types tested here; unsupervised (the Isolation forest), and supervised (ANN).
Some anomalies are easier to detect with the state from earlier layers, while some are easier to detect with the state of later layers.


## Input data
![alt text](data.png)
I use the model state of the MNIST 10-digit classifier model as data for an anomaly detection model. We get the state of the neurons of a specific layer as a CSV file. I also use normal pixel data from MNIST to compare which types of anomalies this standard model is detecting compared to the model-state approach. 

## First method
The first method regarding the isolation forest that I performed is simply generating outbound and inbound mnist data and passing it as an input (model = IsolationForest(max_samples=10, contamination = 0.005)) and checking the accuracy of the model.

## Second method
The second method is to pass the model state data as input. This model state data was firstly obtained by passing mnist inbound and outbound data into a CNN and then by doing: (intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense').output) 





## Technologies
This project is created with:
- Python 3.9.6
- TensorFlow 2.9
- Keras 2.10
- Pandas 1.5.1
- Numpy 1.23.4
- Seaborn 0.12.1
- Matplotlib 
