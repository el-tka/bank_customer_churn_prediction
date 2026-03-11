from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 123


def train_models(features_train, target_train):

    models = [
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        RandomForestClassifier(random_state=RANDOM_STATE),
        LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')
    ]

    for model in models:
        model.fit(features_train, target_train)

    return models


def find_best_random_forest(features_train, target_train, features_valid, target_valid):

    best_model = None
    best_score = 0

    for depth in range(1, 25):

        model = RandomForestClassifier(
            max_depth=depth,
            random_state=RANDOM_STATE
        )

        model.fit(features_train, target_train)

        predictions = model.predict(features_valid)

        score = f1_score(target_valid, predictions)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score


def tune_random_forest_estimators(features_train, target_train, features_valid, target_valid):

    best_model = None
    best_score = 0

    for n in range(5, 100):

        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=8,
            random_state=RANDOM_STATE
        )

        model.fit(features_train, target_train)

        predictions = model.predict(features_valid)

        score = f1_score(target_valid, predictions)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score