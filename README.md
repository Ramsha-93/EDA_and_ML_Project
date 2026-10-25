# EDA_and_ML_Project

This project performs Exploratory Data Analysis (EDA) and compares multiple machine learning models on the Titanic dataset to predict survival probability.

## Project Overview

The Titanic dataset is a classic beginner-friendly classification problem. In this project, we:

- load the Titanic dataset
- clean missing values
- create meaningful features
- preprocess categorical and numerical data
- train multiple machine learning models
- compare model accuracy
- identify the best-performing model

## Objective

The main goal of this project is to build and evaluate a supervised machine learning pipeline that predicts whether a passenger survived or not based on available passenger information.

## Dataset

The project uses the Titanic dataset stored as:

- `titanic.csv`

This file should be placed in the same directory as the Python script before running the project.

## Features Used

The model uses features such as:

- Passenger Class (`Pclass`)
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- FamilySize
- IsAlone
- Title

The target variable is:

- `Survived`

## Data Cleaning Steps

The script includes the following preprocessing steps:

- removes the `Cabin` column due to excessive missing values
- fills missing `Age` values with the median
- fills missing `Embarked` values with the most frequent value
- creates `FamilySize`
- creates `IsAlone`
- extracts `Title` from passenger names
- groups uncommon titles into a single category

## Machine Learning Models

The project compares the following models:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)

Each model is trained using a preprocessing pipeline for numerical and categorical columns.

## Workflow

1. Load the Titanic dataset
2. Clean and transform data
3. Split data into training and testing sets
4. Apply preprocessing
5. Train models
6. Predict on the test set
7. Calculate accuracy
8. Compare results
9. Display the best-performing model

## Requirements

Make sure you have Python installed, then install the required packages:

```bash
pip install pandas numpy scikit-learn
```

## Project Structure

```text
EDA_and_ML_Project/
├── EDA_&_ML.py
├── README.md
├── titanic.csv
```

## How to Run

Open the terminal in the project folder and run:

```bash
python "EDA_&_ML.py"
```

## Expected Output

The script prints:

- accuracy of each model
- final comparison table
- best model name
- best accuracy

## Notes

This project is useful for learning:

- data cleaning
- feature engineering
- preprocessing pipelines
- classification models
- model evaluation

## Summary

This repository demonstrates a complete machine learning workflow on the Titanic dataset, from data preparation to model comparison, making it a strong beginner-friendly project for EDA and ML practice.


