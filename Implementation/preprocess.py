import pandas as pd
import numpy as np
import random


def preprocess(dataset: pd.DataFrame):
    """
    Preprocess dataset to be used for classification
    :param dataset: the given dataset
    :return: train dataset, target labels, data with in_progress label
    """

    # Checking lost values
    print(dataset.isnull().sum())

    # Feature creation -> Product_Mean_Date_Diff (Mean of DateDiff for different products)
    for product in dataset['Product'].unique():
        dataset.loc[(dataset['Product'] == product), 'Product_Mean_Date_Diff'] = dataset['DateDiff'].where(dataset['Product'] == product).sum() / ((dataset['Product'] == product).where(dataset['Deal_Stage'] == 'Won').sum())
    # Feature creation -> Win_Rate (Rate of won per sales agent)
    for agent in dataset['Sales_Agent'].unique():
        dataset.loc[(dataset['Sales_Agent'] == agent), 'Win_Rate'] = ((dataset['Deal_Stage'] == 'Won').where(dataset['Sales_Agent'] == agent).sum() / (dataset['Sales_Agent'] == agent).sum())

    # Modifying column DateDiff to be 1 if DateDiff is less than Product_Mean_Date_Diff or 0 otherwise
    for i in range(0, len(dataset)):
        if dataset.loc[i, 'DateDiff'] < dataset.loc[i, 'Product_Mean_Date_Diff']:
            dataset.loc[i, 'DateDiff'] = 1
        else:
            dataset.loc[i, 'DateDiff'] = 0
    # Renaming DateDiff after modifying
    dataset = dataset.rename(columns={'DateDiff': 'Early_accomplished'})

    # Separating train dataset (won and lost) and in_progress_dataset from main dataset
    train_dataset = dataset.copy()
    in_progress_dataset = train_dataset.loc[train_dataset.Deal_Stage == 'In Progress', train_dataset.columns]
    train_dataset = train_dataset.loc[train_dataset.Deal_Stage != 'In Progress', train_dataset.columns]

    # Resetting indices to work properly in next steps
    train_dataset.reset_index(drop=True, inplace=True)

    # Dropping unnecessary columns
    train_dataset = train_dataset.drop(
        ['Account', 'Opportunity_ID', 'Sales_Agent', 'SalesAgentEmailID', 'ContactEmailID', 'Created Date',
         'Close Date', 'Product_Mean_Date_Diff'], axis=1)
    in_progress_dataset = in_progress_dataset.drop(
        ['Account', 'Opportunity_ID', 'Sales_Agent', 'SalesAgentEmailID', 'ContactEmailID', 'Created Date',
         'Close Date', 'Product_Mean_Date_Diff'], axis=1)

    # Separating targets from train dataset
    targets = train_dataset['Deal_Stage']
    train_dataset = train_dataset.drop(['Deal_Stage'], axis=1)
    in_progress_dataset = in_progress_dataset.drop(['Deal_Stage'], axis=1)

    # One-Hot encoding the product feature as a categorical data
    train_dataset = train_dataset.join(pd.get_dummies(train_dataset['Product'])).drop('Product', axis=1)
    in_progress_dataset = in_progress_dataset.join(pd.get_dummies(in_progress_dataset['Product'])).drop('Product',
                                                                                                        axis=1)

    # Removing Outliers
    from scipy import stats
    z_scores = stats.zscore(train_dataset)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    train_dataset = train_dataset[filtered_entries]
    targets = targets[filtered_entries]

    # Balancing data:
    # Removing additional Won labeled data in random
    size = len(train_dataset) - 1
    won_indices = []
    won_cnt = 0
    remove_indices = []
    for i, label in targets.items():
        if label == 'Won':
            won_indices.append(i)
            won_cnt += 1
    lost_cnt = size - won_cnt
    for i in range((won_cnt - lost_cnt)):
        r_index = random.randint(0, won_cnt - i - 1)
        remove_indices.append(won_indices[r_index])
        won_indices.pop(r_index)

    train_dataset.drop(remove_indices, inplace=True, axis=0)
    targets.drop(remove_indices, inplace=True, axis=0)

    return train_dataset, targets, in_progress_dataset
