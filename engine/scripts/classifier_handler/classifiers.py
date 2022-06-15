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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Classifier:
    def __init__(self, weights):
        self.weights = weights
        self.use_pca = PERFORM_PCA_ANALYSIS
        self.model = None
        self.name = None
        self.pca = None
        self.predict_probability = True

    def fit(self, data):
        if self.use_pca:
            data = self.perform_pca(data, True)
        target = data['match']
        inputs = data.drop(columns=['match'])

        if EQUALIZE_CLASS_IMPORTANCE:
            value_counts = target.value_counts()
            positive_sample_count = POSITIVE_CLASS_UPSAMPLING_RATIO * value_counts.loc[0] / value_counts.loc[1]
            sample_weights = target.apply(lambda match: positive_sample_count if match == 1 else 1)
            self.model.fit(inputs, target, sample_weight=sample_weights)
        else:
            self.model.fit(inputs, target)

    def predict(self, data, predict_outputs=True):
        if self.use_pca:
            data = self.perform_pca(data, False)
        if 'match' in data.columns:
            data = data.drop(columns=['match'])

        if self.predict_probability:
            scores = self.model.predict_proba(data)
            scores = [s[1] for s in scores]
        else:
            scores = self.model.predict(data)

        if predict_outputs:
            outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
            return outputs, scores
        else:
            return scores

    def print_feature_importance(self, feature_names):
        pass

    def set_threshold(self, threshold):
        self.weights['threshold'] = threshold

    def save(self, path='results/models', key_value_store=None):
        if key_value_store is None:
            if not os.path.exists(path):
                os.makedirs(path)

            model_filename = f'{self.name}_model.sav'
            weights_filename = f'{self.name}_weights.sav'

            pickle.dump(self.model, open(os.path.join(path, model_filename), 'wb'))
            pickle.dump(self.weights, open(os.path.join(path, weights_filename), 'wb'))
            if self.use_pca:
                pca_filename = f'{self.name}_pca.sav'
                pickle.dump(self.pca, open(os.path.join(path, pca_filename), 'wb'))
        else:
            key_value_store.set_record('model', pickle.dumps(self.model))
            key_value_store.set_record('weights', pickle.dumps(self.weights))
            if self.use_pca:
                key_value_store.set_record('pca', pickle.dumps(self.pca))

    def load(self, path='results/models', key_value_store=None):
        if key_value_store is None:
            model_filename = f'{self.name}_model.sav'
            weights_filename = f'{self.name}_weights.sav'
            self.model = pickle.load(open(os.path.join(path, model_filename), 'rb'))
            self.weights = pickle.load(open(os.path.join(path, weights_filename), 'rb'))
            if self.use_pca:
                pca_filename = f'{self.name}_pca.sav'
                self.pca = pickle.load(open(os.path.join(path, pca_filename), 'rb'))
        else:
            self.model = pickle.loads(key_value_store.get_record('model')['value'])
            self.weights = pickle.loads(key_value_store.get_record('weights')['value'])
            if self.use_pca:
                self.pca = pickle.loads(key_value_store.get_record('pca')['value'])

    def perform_pca(self, data, train_pca=False):
        principal_component_count = PRINCIPAL_COMPONENT_COUNT
        data_match = pd.Series()
        if 'match' in data:
            data_match = data['match']
            data = data.drop(columns='match')

        if train_pca:
            self.pca = PCA(n_components=principal_component_count)
            data = self.pca.fit_transform(data)
        else:
            data = self.pca.transform(data)

        data_principal_components = pd.DataFrame(data,
                                                 columns=["principal_component_{}".format(component) for component in
                                                          range(1, principal_component_count + 1)])
        if not len(data_match) == 0:
            data_principal_components['match'] = data_match

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


class NeuralNetworkClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = MLPClassifier(**parameters)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(
            f'Number of layers: {self.model.n_layers_} and their shapes; '
            f'{len(self.model.coefs_[0])}, {self.model.hidden_layer_sizes}, {self.model.n_outputs_}'
        )
        print(f'Feature importance for {self.name} \n')
        for i, weights in enumerate(self.model.coefs_):
            weights = [[round(w, 4) for w in weight] for weight in weights]
            print(f'Layer {i} to {i + 1}: \n {weights}')


class EnsembleModellingClassifier(Classifier):
    def __init__(self, weights, parameters):
        super().__init__(weights)
        self.model = []
        self.name = 'EnsembleModellingClassifier'
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
        if self.use_pca:
            data = self.perform_pca(data, False)
        outputs_array = []
        scores_array = []
        for classifier in self.model:
            if 'match' in data.columns:
                data = data.drop(columns=['match'])

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
        return outputs, scores
