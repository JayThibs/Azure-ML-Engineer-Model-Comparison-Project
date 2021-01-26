# Optimizing an ML Pipeline in Azure

## Overview

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model (Logistic Regression). To find the optimal hyperparameters for the Logistic Regression model, we use AzureML's HyperDrive which helps use perform a hyperparameter search. In other words, we train many models on our training data to see how the hyperparameters have an effect on performance and find the best hyperparameters.

This model is then compared to an Azure AutoML run.

* [train.py](https://github.com/JayThibs/Azure-ML-Engineer-Model-Comparison-Project/blob/main/train.py): Script containing the `clean_data` function as well as the code for training the logistic regression model.

* [hyperdrive_vs_automl_comparison.ipynb](https://github.com/JayThibs/Azure-ML-Engineer-Model-Comparison-Project/blob/main/hyperdrive_vs_automl_comparison.ipynb): Notebook containing the code that uses the AzureML python SDK to create experiments, set dependencies, spin up computer cluster, submit experiments, and save the best model.

## Summary
This dataset contains data regarding direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit, based on features such as age, job, marital status, and education. Dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing.

The HyperDrive model had an accuracy of 0.9112, while **the best AutoML model (VotingEnsemble) had an accuracy of 0.9166**. Therefore, the **AutoML model performed slightly better** than the Logistic Regression model trained with HyperDrive.

## Scikit-learn Pipeline
### Explanation of the pipeline architecture:

The pipeline is broken up into 7 steps.

1. Import the data with `TabularDatasetFactory` with the following URL: https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv.
2. Cleaning and one hot encoding the data.
3. Split the data into a train and test set.
4. Create the model object with the hyperparameters we will be testing.
5. Fit the model on our training data.
6. Test model accuracy with the testing data.
7. Save (register) the best model.

* Steps 1-6 can be found in [train.py](https://github.com/JayThibs/Azure-ML-Engineer-Model-Comparison-Project/blob/main/train.py).
* Step 7 model registration for the **best model** is found in the [notebook](https://github.com/JayThibs/Azure-ML-Engineer-Model-Comparison-Project/blob/main/udacity-project.ipynb), after the experiment has been submitted and the model has been trained. However, the pkl file is first created in [train.py](https://github.com/JayThibs/Azure-ML-Engineer-Model-Comparison-Project/blob/main/train.py), which saves all the other models with different hyperparameters.

For the HyperDrive model, we used a [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model since it is a simple model that works well with binary classification. With HyperDrive, we are looking to optimize the following logistic regression model hyperparameters: `C` and `max_iter`.

`C`: The `C` parameter controls the penalty (regularization) strength, which helps the model avoid overfitting. Smaller values specify stronger regularization. We used continuous values from 0.001 to 1.0.

`max_iter`: Maximum number of iterations taken for the model to converge. We used the following values: [10, 50, 100].

### Benefits of the parameter sampler we chose:

We used the `RandomParameterSampling` (Random Search) method to test the performance of different parameters for our logistic regression model. For Random Search, you define a search space as a bounded domain of hyperparameter values and randomly sample points in that domain. In contrast, Grid Search you specify and evaluate every point within the space.

Random Search is generally better than Grid Search because it allows you to achieve the same performance (if not, it's close), but with much less training runs with different hyperparameters.

### The early stopping policy we chose:

We used the [Bandit Policy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) for our early stopping policy. Bandit policy stops a run if the target performance metric doesn't fall within the `slack factor` (ratio) or `slack amount` (absolute amount) of the best run so far. In other words, this policy allows us to end runs when a model is no longer improving so that we may move on to training the next model. This helps save time and computation resources.

We chose an `evaluation interval` of 2 (the policy is applied every two interval where the primary metric is reported).

We also chose a `slack factor` of 0.1. Any run whose accuracy is less than (1 / (1 + 0.1)) or 91% of the best run will be terminated.

## AutoML

AutoML allows you to automatically run a series of different algorithms and parameters for you. In our experiment, there were 32 models trained, and VotingEnsemble had the best accuracy at 91.66%.

The hyperparameters for the model:

* `reg_alpha=0` (default value)
* `reg_lambda=0.520833`
* `scale_pos_weight=1` (default value)
* `seed=None` (default value)
* `silent=None`
* `subsample=0.6`
* `tree_method='auto'` (default value)
* `verbose=-10`
* `verbosity=0`

## Pipeline Comparison

* HyperDrive Logistic Regression model accuracy: 91.12%
* AutoML VotingEnsemble model accuracy: 91.66%
* Difference in accuracy: 0.54%

AutoML tested multiple algorithms (including ensembles), while HyperDrive only used Logistic Regression. The benefit of AutoML is that you are able to test many algorithms quickly, and therefore get a high accuracy without much work. However, it would not have required much more work to also test out other algorithms. I expect we could have beaten AutoML had we used another algorithm (like gradient boosted trees) with HyperDrive.

AutoML automatically chose a set of hyperparameters for each model trained. For HyperDrive, we explicitly chose a domain of values that were used for training the model.

## Future Work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

### Different Algorithms

HyperDrive should be tested with other algorithms that tend to get higher accuracy than logistic regression (gradient boosted trees). We coult even train multiple models and then create an ensemble, like what was done with AutoML.

### Different Datasets

For this project, we only tested for one specific dataset, but it would be interesting to test how well AutoML would perform against HyperDrive if we had different types of datasets.

### Early Stopping Policies

We could extend the 30 minute timeout of the AutoML approach to see if we can discover new models with higher accuracy.

### Bayesian Optimization

We would like to try using Bayesian Optimization to train our HyperDrive model.

### Dealing with Data Imbalance

Our output of the AutoML approach said that there was a possible data imbalance in our dataset. If we want to improve performance of our models, it may be worth it to deal with that issue and balance out the classes.

## Cluster clean up

The final cell of the Jupyter Notebook deletes the computer cluster created for training the models.
