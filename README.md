# Disaster Classification - NLP

This project demonstrates how to classify text data into "disaster" or "not disaster" categories using a Naive Bayes classifier with TF-IDF vectorization. The goal is to predict whether a given sentence is related to a disaster or not based on historical data.

---

## Dataset Visualization:
<img src="https://github.com/leovidith/DisasterDetection-NLP/blob/main/images/kde.png"  width="600">
<img src="https://github.com/leovidith/DisasterDetection-NLP/blob/main/images/pie%20charts.png"  width="600">

---

## Table of Contents

1. [Requirements](#requirements)
2. [Data](#data)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Evaluation](#model-evaluation)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow

## Data

The project uses two datasets:

1. `train.csv`: The training dataset containing text and labels.
2. `test.csv`: The test dataset for evaluation.

The datasets are downloaded from the following sources:
- [Training and Test Data](https://colab.research.google.com/drive/1zpmkxIU_e4O0FP67F3fsQjJDjnmxjqpI)

## Installation

1. Clone the repository or download the project files.

2. Install the required Python packages:
    ```bash
    pip install pandas numpy scikit-learn tensorflow
    ```

3. Download and unzip the data:
    ```bash
    wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
    wget https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip
    python helper_functions.py unzip_data("nlp_getting_started.zip")
    ```

## Usage

1. **Run the script**:
   Execute the script to train the model and make predictions:
   ```bash
   python disaster_classification.py
   ```

2. **Provide Input**:
   You will be prompted to enter a sentence. The model will classify it as either "Disaster" or "Not disaster" based on the training data.

   For example:
   ```
   Enter a sentence: floods 
   Model Prediction: Disaster
   ```

## Model Evaluation

The model's performance has been evaluated with the following metrics:

- **Accuracy**: 81.10%
- **Precision**: 81.10%
- **Recall**: 81.10%
- **F1 Score**: 80.46%

These metrics indicate that the model performs well, with balanced precision and recall, and a high overall accuracy.
