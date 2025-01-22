# EthioMart NER Project

## Overview
This repository contains the code and resources for fine-tuning a Named Entity Recognition (NER) model to extract key entities like products, prices, and locations from Amharic Telegram messages. The project involves data preprocessing, model training, and evaluation, with the fine-tuning process conducted using Google Colab.

## Directory Structure
├── notebooks/ # Contains Jupyter notebooks for demo and model training │   
├── scripts/ # Python scripts for preprocessing │  
       
├── scraping.py # Script for scraping data from Telegram channels │    
├── requirements.txt # List of required Python packages   
├── README.md # Project documentation   
└── .gitignore # Files to ignore in the Git repo


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/senaygui/ethiomart.git
cd ethiomart
```

### 2. Install dependencies
Ensure that you have Python 3.7 or higher installed. Install the required packages by running:

```bash
pip install -r requirements.txt
```

### 3. Directory Details
notebooks/: This directory contains Jupyter notebooks demonstrating preprocessing and the fine-tuning process.

preprocessing.ipynb: Demonstrates the dataset preprocessing using the preprocessing.py script.  
fine_tuning.ipynb: This notebook handles fine-tuning the NER model. It includes data loading, tokenization, model training, and evaluation. This notebook was run on Google Colab to take advantage of GPU resources for faster training.  
scripts/: Contains Python scripts for scraping and preprocessing tasks.

scraping.py: A Python script for scraping messages from multiple Ethiopian Telegram e-commerce channels. It fetches text, images, and metadata (e.g., timestamps, sender info) in real-time and stores them for preprocessing.
preprocessing.py: A Python script for cleaning, tokenizing, and preparing raw text data for NER model training, as well as combining datasets.


requirements.txt: A list of all the Python libraries and versions used in the project, including transformers, datasets, torch, numpy, and more.

### 4. Running the Project
To run the project, you can follow these steps:

Fine-Tuning: Open and run fine_tuning.ipynb on Google Colab. Make sure to switch to a GPU runtime before running the notebook to speed up the fine-tuning process.

Model Training and Evaluation: In fine_tuning.ipynb, the model is fine-tuned on the preprocessed dataset, and evaluation metrics such as precision, recall, and F1-score are calculated for each entity type (Product, Price, Location).

### 5. Model Evaluation
The fine-tuned models are evaluated using the following metrics:

Precision  
Recall  
F1-Score  
Overall Accuracy  
These metrics are computed for each entity type (Product, Price, Location), along with the overall performance of the model.

Additional Notes  
Google Colab: The fine_tuning.ipynb notebook was executed using Google Colab for GPU acceleration. Ensure you select the GPU runtime before training the model to optimize performance.

Model Interpretability: Future updates may include model interpretability tools (e.g., SHAP, LIME) to provide explanations for the model's entity predictions and increase transparency.