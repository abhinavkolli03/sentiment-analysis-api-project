import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# class for all binary classifier ML models implemented
class Models:
    def __init__(self):
        print("Setting pre-trained and new models...")
        self.models = []
        self.create_estimators()

    def create_estimators(self):
        # Logistic Regression
        params = [
            {'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
            {'penalty': ['l2'], 'solver': ['liblinear', 'sag', 'saga']}
        ]
        self.models.append((LogisticRegression(), params))

        # KNN Classifier
        params = {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19],
            "leaf_size": [1, 5, 10, 15, 30]
        }
        self.models.append((KNeighborsClassifier(), params))

        # Multinomial Naive Bayes
        params = {
            "alpha": np.linspace(0.3, 1.3, 10)
        }
        self.models.append((MultinomialNB(), params))

        # Linear Support Vector Classifier
        params = {
            "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1]
        }
        self.models.append((LinearSVC(), params))

        # Decision Tree Classifier
        params = {
            "min_samples_split": np.linspace(0.1, 1, 9),
            "max_features": ["sqrt", "log2"],
            "criterion": ["entropy", "gini"]
        }
        self.models.append((DecisionTreeClassifier(), params))

        # Random Forest Classifier
        params = {
            "n_estimators": [64, 100, 128, 200],
            "criterion": ["entropy", "gini"],
            "max_features": ["log2", "sqrt"],
            "max_depth": np.linspace(3, 8, 5)
        }
        self.models.append((RandomForestClassifier(), params))

        # Gradient Boost Classifier
        params = {
            "loss": ["deviance", "exponential"],
            "n_estimators": [64, 100, 128, 200],
            "learning_rate": [0.1, 0.05, 0.2],
            "max_depth": np.linspace(3, 6, 3),
            "max_features": ["log2", "sqrt"]
        }
        self.models.append((GradientBoostingClassifier(), params))

        return self.models
        # you can add more pre-trained binary classifiers here

    def train_grid(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        #StraitifiedKFold will help with splitting test/train sets against unbalanced positive/negative classes
        self.cv = StratifiedKFold(shuffle=True)
        estimators = list()
        for model, params in self.models:
            print("Model: ", model.__class__.__name__)
            grid_model = GridSearchCV(model, param_grid=params, cv=self.cv)
            print("Fitting model...")
            grid_model.fit(self.X_train, self.y_train)
            print("Grid search's best estimator for the model...")
            print(grid_model.best_estimator_)
            print('\n')
            #store estimators for testing phase
            estimators.append(grid_model)
            joblib.dump(grid_model, f"../models/{model.__class__.__name__}.pkl")
        return estimators

    def test_grid(self, estimators, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        for estimator in estimators:
            #goes through the list of models and performs accuracy results
            predictions = estimator.predict(X_test)
            print("Model performance results...")
            print(f"Accuracy Score: {accuracy_score(predictions, y_test) * 100}%")
            print(classification_report(predictions, y_test))
            plot_confusion_matrix(estimator, X_test, y_test)
            print("\n")

    def infer(self, model, features):
        #test model after API call and return prediction from pre-trained model
        predictions = model.predict(features)
        return predictions