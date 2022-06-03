# Non-invasive blood pressure prediction from PPG signals using deep learning

## Description

This project is my master's thesis work which contains code on the 
proposed deep neural network, using a convolutional neural network(CNN)
and a long short-term memory (LSTM) model to predict three parameters: 
diastolic blood pressure (DBP), systolic blood pressure (SBP), and heart rate (HR).


### Project
Trainings are performed using PyTorch 1.10.2 and Python 3.9 frameworks. 

- **analyse/data_analyse.py** file consists of script to view the single patient record dataset and data analysis.


- The folder **extract_preprocessing** contains the scripts used for downloading data (physiological raw PPG 
and ground truth labels -SBP, DBP, and HR), dividing signals into windows and preprocessing methods.\
PPG and ABP signals are extracted from each record and divided 
into windows of a defined length and overlap. Several preprocessing 
steps include removing empty records, unreliable BP values, outliers, filtering the PPG signal 
and finally normalizing the signals. 


- The folder **model** composed scripts on model developement (proposed neural 
network architecture) and trains neural networks for BP prediction
using PPG data; saves the trained model for later fine tuning and 
personalization using PPG data.\
To train the network, the dataset is divided into training, validation and test set.\
The proposed deep learning framework involves CNN and LSTM model to estimate the patient’s instantaneous
Heart Rate, Systolic and Diastolic blood pressure.



### Datasource
MIMIC-III (Medical Information Mart for Intensive Care) freely
available database was used for extracting the data.\
https://physionet.org/content/mimic3wdb/1.0/

![alt-text]("/Users/nikitha/Desktop/thesis_pics/Figure 3.1.png "optional-title")




#### Setup

Clone for Github repository, or can download the repository and unzip it.\
https://github.com/Nikitha-ramasetti/BP_Estimation_PPG

```sh
git clone <repo_url>
```


#### Installation
To create a virtual environment using Python 3.9 as interpreter,
 the virtualenv package is required. 
 
```sh
pip install virtualenv
```

The virtual environment can then be created using the command.
```sh
virtualenv --python=/usr/bin/python3.9 venv/
```


The virtual environment can be activated using the command.
```sh
source venv/bin/activate
```

Necessary python packages can be installed using the
command from the **requirements.txt** file.

```sh
pip install -r requirements.txt
```



