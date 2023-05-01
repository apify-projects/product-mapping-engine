import os
import pickle
from io import StringIO

import numpy as np
import pandas as pd
import pydot
from configuration import PRINCIPAL_COMPONENT_COUNT, POSITIVE_CLASS_UPSAMPLING_RATIO, EQUALIZE_CLASS_IMPORTANCE, \
    PERFORM_PCA_ANALYSIS
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RandomForests
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import GradientBoostingClassifier as GradientBoosting
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Classifier:
    def __init__(self, weights=[]):
        self.weights = weights
        self.use_pca = PERFORM_PCA_ANALYSIS
        self.model = None
        self.name = None
        self.pca = None
        self.predict_probability = True

    def fit(self, data):
        if EQUALIZE_CLASS_IMPORTANCE:
            matching_data = data[data['match'] == 1]
            matching_data_ratio = len(data[data['match'] == 0]) / len(matching_data)
            print(f'Recommended upsampling of matches is {matching_data_ratio}')
            print(f'Used upsampling of matches is {POSITIVE_CLASS_UPSAMPLING_RATIO}')
            data = data.append([matching_data] * (POSITIVE_CLASS_UPSAMPLING_RATIO - 1),
                                           ignore_index=True)
            data = data.sample(frac=1)

        target = data['match']
        inputs = data.drop(columns=['match'])

        if self.use_pca:
            inputs = self.perform_pca(inputs, True)

        self.model.fit(inputs, target)

    def predict(self, data, predict_outputs=True):
        inputs = data
        if 'match' in inputs.columns:
            inputs = inputs.drop(columns=['match'])

        if self.use_pca:
            inputs = self.perform_pca(inputs, False)

        if self.predict_probability:
            scores = self.model.predict_proba(inputs)
            scores = [s[1] for s in scores]
        else:
            scores = self.model.predict(inputs)

        if predict_outputs:
            outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
            return outputs, scores
        else:
            return scores

    def print_feature_importance(self, feature_names):
        pass

    def set_threshold(self, threshold):
        self.weights['threshold'] = threshold

    def perform_pca(self, data, train_pca=False):
        principal_component_count = PRINCIPAL_COMPONENT_COUNT

        if train_pca:
            self.pca = PCA(n_components=principal_component_count)
            data = self.pca.fit_transform(data)
        else:
            data = self.pca.transform(data)

        data_principal_components = pd.DataFrame(
            data,
            columns=[
                "principal_component_{}".format(component) for component in range(1, principal_component_count + 1)
            ]
        )

        return data_principal_components


class LinearRegressionClassifier(Classifier):
    def __init__(self, weights, _):
        super().__init__(weights)
        self.model = LinearRegression()
        self.name = str(type(self.model)).split(".")[-1][:-2]
        self.predict_probability = False

    def predict(self, data, predict_outputs=True):
        scores = self.model.predict(data.drop(columns=['match']))
        if predict_outputs:
            outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
            return outputs, scores
        else:
            return scores

    def print_feature_importance(self, feature_names):
        print(
            f'Feature importance for {self.name} \n '
            f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.coef_]))}'
        )


class LogisticRegressionClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = LogisticRegression(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(
            f'Feature importance for {self.name} \n '
            f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.coef_[0]]))}'
        )


class SupportVectorMachineClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = svm.SVC(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]


class DecisionTreeClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = DecisionTree(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(
            f'Feature importance for {self.name} \n '
            f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.feature_importances_]))}'
        )
        dot_data = StringIO()
        tree.export_graphviz(self.model, out_file=dot_data, filled=True, rounded=True, special_characters=False,
                             impurity=False, feature_names=feature_names)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("decision_tree.pdf")


class RandomForestsClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = RandomForests(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names, visualize=False):
        print(f'Feature importance for {self.name} \n '
              f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.feature_importances_]))}')
        if visualize:
            if not os.path.exists('random_forest_visualization'):
                os.makedirs('random_forest_visualization')
            for i, one_tree in enumerate(self.model.estimators_):
                dot_data = StringIO()
                tree.export_graphviz(one_tree, out_file=dot_data, filled=True, rounded=True, special_characters=False,
                                     impurity=False, feature_names=feature_names)
                graph = pydot.graph_from_dot_data(dot_data.getvalue())
                graph[0].write_pdf(f"random_forest_visualization/random_forests_{i}.pdf")

class EnsembleClassifier(Classifier):
    pass

class NeuralNetworkClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = MLPClassifier(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        pass


class BaggingClassifier(EnsembleClassifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = []
        self.name = 'BaggingClassifier'
        for classifier_type in parameters:
            for model_params in parameters[classifier_type]:
                classifier_class_name = classifier_type + 'Classifier'
                classifier_class = getattr(
                    __import__('classifier_handler.classifiers', fromlist=[classifier_class_name]),
                    classifier_class_name)
                classifier = classifier_class({}, model_params)
                self.model.append(classifier)

    @staticmethod
    def combine_predictions_from_classifiers(predicted_values, combination_type):
        predicted_values = np.array(predicted_values)
        if combination_type == 'score':
            return np.mean(predicted_values, axis=0)
        else:
            predicted_values = np.mean(predicted_values, axis=0)
            return [1 if output >= 0.5 else 0 for output in predicted_values]

    def predict(self, data, predict_outputs=True):
        if 'match' in data.columns:
            data = data.drop(columns=['match'])

        if self.use_pca:
            data = self.perform_pca(data, False)
        outputs_array = []
        scores_array = []

        for classifier in self.model:
            if self.predict_probability:
                scores = classifier.model.predict_proba(data)
                scores = [s[1] for s in scores]
            else:
                scores = classifier.model.predict(data)

            if predict_outputs:
                outputs_array.append([0 if score < self.weights['threshold'] else 1 for score in scores])
            scores_array.append(scores)
        outputs = self.combine_predictions_from_classifiers(outputs_array, 'output')
        scores = self.combine_predictions_from_classifiers(scores_array, 'score')
        #outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
        return outputs, scores


class BoostingClassifier(BaggingClassifier):
    def __init__(self, weights, parameters):
        super().__init__(weights, parameters)
        self.name = 'BoostingClassifier'


class AdaBoostClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = AdaBoost(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]


class GradientBoostingClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = GradientBoosting(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]