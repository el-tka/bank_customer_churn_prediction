import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

RANDOM_STATE = 123


def load_data(path):
    """Load dataset"""
    data = pd.read_csv(path)
    return data


def preprocess_data(data):
    """Basic preprocessing"""

    # fill missing values
    data['Tenure'] = data['Tenure'].fillna(data['Tenure'].mean())

    # target and features
    target = data['Exited']
    features = data.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)

    # one hot encoding
    features = pd.get_dummies(features)

    return features, target


def split_data(features, target):

    features_train, features_temp, target_train, target_temp = train_test_split(
        features, target, test_size=0.4, random_state=RANDOM_STATE)

    features_valid, features_test, target_valid, target_test = train_test_split(
        features_temp, target_temp, test_size=0.5, random_state=RANDOM_STATE)

    return features_train, features_valid, features_test, target_train, target_valid, target_test


def scale_data(features_train, features_valid, features_test):

    numeric = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    scaler = StandardScaler()
    scaler.fit(features_train[numeric])

    features_train[numeric] = scaler.transform(features_train[numeric])
    features_valid[numeric] = scaler.transform(features_valid[numeric])
    features_test[numeric] = scaler.transform(features_test[numeric])

    return features_train, features_valid, features_test


def upsample(features, target, repeat):

    features_zeros = features[target == 0]
    features_ones = features[target == 1]

    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled,
        target_upsampled,
        random_state=RANDOM_STATE
    )

    return features_upsampled, target_upsampled