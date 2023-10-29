# EPFL CS-433 Machine Learning
## Project 1: Model Building and Validation for MICHD Prediction

The goal of the project is to build a model that estimate the risk of developing cardiovascular diseases (CVD) in terms of health-related data and lifestyle of a person. The data used for this project is from https://www.cdc.gov/brfss/annual_data/annual_2015.html .

### Getting Started

The project is written in python 3.12.0 and the following libraries are required to run the code:
* numpy 1.26.1
* matplotlib 3.8.0

### Project Structure

The project is structured as follows:
* `data` folder contains the data used for the project
* `results` folder contains the results of the project
* `src` folder contains the source code of the project
  - `implementations.py` contains the functions used to process the data, build the model and output the results
  - `run.ipynb` contains the code to run the project

### Running the project

To run the project, open the `run.ipynb` notebook and run the cells in order. The notebook will output the results in the `results` folder. 
By setting the global variables "NN_METHOD" and "PCA", the notebook can run different dataprocessing methods for different models. In general, we apply normalization method to the NN model without PCA, while we apply standardization method to the other models with PCA. The comparative analysis of different methods is shown in the report.

### Authors

* **Yihan Wang** - *
* **Zewei Zhang** - *