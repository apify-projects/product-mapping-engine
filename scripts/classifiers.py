import os
import pickle
from io import StringIO

import pandas as pd
import pydot
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RandomForests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from configuration import PRINCIPAL_COMPONENT_COUNT, PERFORM_PCA_ANALYSIS
from evaluate_classifier import plot_train_test_roc

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Classifier:
    def __init__(self, weights):
        self.weights = weights
        self.use_pca = PERFORM_PCA_ANALYSIS
        self.model = None
        self.name = None
        self.pca = None

    def fit(self, data):
        if self.use_pca:
            data = self.perform_pca(data, True)
        target = data['match']
        inputs = data.drop(columns=['match'])
        self.model.fit(inputs, target)

    def predict(self, data, predict_outputs=True):
        if self.use_pca:
            data = self.perform_pca(data, False)
        if 'match' in data.columns:
            data = data.drop(columns=['match'])
        scores = self.model.predict_proba(data)
        scores = [s[1] for s in scores]
        if predict_outputs:
            outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
            return outputs, scores
        else:
            return scores

    def render_roc(self, true_labels, predicted_labels_list, threshes, print_stats):
        plot_train_test_roc(true_labels, predicted_labels_list, threshes, print_stats)

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

    def dataframe_pca_results(self, pca_result, auxiliary_columns, auxiliary_data, principal_component_count):
        dataframe = pd.DataFrame(pca_result, columns=["principal_component_{}".format(component) for component in
                                                      range(1, principal_component_count + 1)])
        auxiliary_data = auxiliary_data.reset_index(drop=True)
        for column in auxiliary_columns:
            dataframe[column] = auxiliary_data[column]
        return dataframe

    def perform_pca(self, data, train_pca=False):
        principal_component_count = PRINCIPAL_COMPONENT_COUNT

        auxiliary_columns = ["index1", "index2"]
        if "match" in data:
            auxiliary_columns.append("match")
        data_auxiliary_columns = data[auxiliary_columns]
        data = data.drop(columns=auxiliary_columns)

        if train_pca:
            self.pca = PCA(n_components=principal_component_count)
            data = self.pca.fit_transform(data)
        else:
            data = self.pca.transform(data)

        data = self.dataframe_pca_results(data, auxiliary_columns, data_auxiliary_columns, principal_component_count)

        return data


class LinearRegressionClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = LinearRegression()
        self.name = str(type(self.model)).split(".")[-1][:-2]

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
    def __init__(self, weights):
        super().__init__(weights)
        self.model = LogisticRegression()
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(
            f'Feature importance for {self.name} \n '
            f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.coef_[0]]))}'
        )


class SvmLinearClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'linear'
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(f'Feature importance for {self.name} \n {dict(zip(feature_names, self.model.coef_[0]))}')


class SvmRbfClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'rbf'
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]


class SvmPolyClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'poly'
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]


class NeuralNetworkClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', solver='adam', max_iter=1000)
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


class DecisionTreeClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = DecisionTree(max_depth=5, max_leaf_nodes=20)
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
    def __init__(self, weights):
        super().__init__(weights)
        self.model = RandomForests(max_depth=5, n_estimators=3)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importance(self, feature_names):
        print(f'Feature importance for {self.name} \n '
              f'{dict(zip(feature_names, ["{:.6f}".format(x) for x in self.model.feature_importances_]))}')
        for i, one_tree in enumerate(self.model.estimators_):
            dot_data = StringIO()
            tree.export_graphviz(one_tree, out_file=dot_data, filled=True, rounded=True, special_characters=False,
                                 impurity=False, feature_names=feature_names)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph[0].write_pdf(f"random_forest_visualization/random_forests_{i}.pdf")
