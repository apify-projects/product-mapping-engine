import os
import pickle
from io import StringIO

import pandas as pd
import pydot
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RandomForests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.decomposition import PCA

from evaluate_classifier import plot_train_test_roc

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Classifier:
    def __init__(self, weights, use_pca=False):
        self.weights = weights
        self.use_pca = use_pca

        # TODO arbitrarily chosen, fix this
        if 'threshold' not in self.weights:
            self.weights['threshold'] = 0.75

        self.model = None
        self.name = None
        self.pca = None

    def fit(self, data):
        if self.use_pca:
            data = self.perform_pca(data, True)
        target = data['match']
        inputs = data.drop(columns=['match'])
        self.model.fit(inputs, target)

    def predict(self, data):
        if self.use_pca:
            data = self.perform_pca(data, False)
        if 'match' in data.columns:
            data = data.drop(columns=['match'])
        scores = self.model.predict_proba(data)
        scores = [s[1] for s in scores]
        outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
        return outputs, scores

    def render_roc(self, true_labels, pred_labels_list, threshs, print_stats):
        plot_train_test_roc(true_labels, pred_labels_list, threshs, print_stats)

    def print_feature_importances(self):
        pass

    def save(self, path='results/models', key_value_store=None):
        # TODO save PCA locally
        if key_value_store is None:
            if not os.path.exists(path):
                os.makedirs(path)

            filename = f'{self.name}.sav'
            pickle.dump(self.model, open(os.path.join(path, filename), 'wb'))
        else:
            key_value_store.set_record('model', pickle.dumps(self.model))
            if self.use_pca:
                key_value_store.set_record('pca', pickle.dumps(self.pca))

    def load(self, path='results/models', key_value_store=None):
        #TODO load PCA locally
        if key_value_store is None:
            filename = f'{self.name}.sav'
            self.model = pickle.load(open(os.path.join(path, filename), 'rb'))
        else:
            self.model = pickle.loads(key_value_store.get_record('model')['value'])
            if self.use_pca:
                self.pca = pickle.loads(key_value_store.get_record('pca')['value'])

    def dataframe_pca_results(self, pca_result, auxiliary_columns, auxiliary_data, principal_component_count):
        dataframe = pd.DataFrame(pca_result, columns=["principal_component_{}".format(component) for component in range(1, principal_component_count+1)])
        auxiliary_data = auxiliary_data.reset_index(drop=True)
        for column in auxiliary_columns:
            dataframe[column] = auxiliary_data[column]
        return dataframe

    def perform_pca(self, data, train_pca=False):
        principal_component_count = 33

        auxiliary_columns = ["index1", "index2"]
        if "match" in data:
            auxiliary_columns.append("match")
            print(auxiliary_columns)
        data_auxiliary_columns = data[auxiliary_columns]
        data = data.drop(columns=auxiliary_columns)

        print(data.columns)

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

    def predict(self, data):
        scores = self.model.predict(data.drop(columns=['match']))
        outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
        return outputs, scores

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.coef_}')


class LogisticRegressionClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = LogisticRegression()
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.coef_}')


class SvmLinearClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'poly'  # linear, rbf, poly
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.coef_}')


class SvmRbfClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'rbf'
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Support vectors for {self.name} \n {self.model.support_vectors_}')


class SvmPolyClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.kernel = 'poly'
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Support vectors for {self.name} \n {self.model.support_vectors_}')


class NeuralNetworkClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', solver='adam', max_iter=1000)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(
            f'Number of layers: {self.model.n_layers_} and their shapes; {len(self.model.coefs_[0])}, {self.model.hidden_layer_sizes}, {self.model.n_outputs_}')
        print(f'Feature importances for {self.name} \n')
        for i, weights in enumerate(self.model.coefs_):
            weights = [[round(w, 4) for w in weight] for weight in weights]
            print(f'Layer {i} to {i + 1}: \n {weights}')


class DecisionTreeClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = DecisionTree(max_depth=5, max_leaf_nodes=20)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.feature_importances_}')
        dot_data = StringIO()
        tree.export_graphviz(self.model, out_file=dot_data, filled=True, rounded=True, special_characters=False,
                             impurity=False, feature_names=list(self.weights.keys())[:-1])
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("results/decision_tree.pdf")


class RandomForestsClassifier(Classifier):
    def __init__(self, weights):
        super().__init__(weights)
        self.model = RandomForests(max_depth=5, n_estimators=3)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.feature_importances_}')
        for i, one_tree in enumerate(self.model.estimators_):
            dot_data = StringIO()
            tree.export_graphviz(one_tree, out_file=dot_data, filled=True, rounded=True, special_characters=False,
                                 impurity=False, feature_names=list(self.weights.keys())[:-1])
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph[0].write_pdf(f"results/random_forests_{i}.pdf")
