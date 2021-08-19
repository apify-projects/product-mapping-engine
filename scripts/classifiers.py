import os
from io import StringIO

import pydot
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RandomForests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from evaluate_classifier import plot_roc

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Classifier:
    def __init__(self, weights):
        self.weights = weights
        self.model = None
        self.name = None

    def fit(self, data):
        target = data['match']
        inputs = data.drop(columns=['match'])
        self.model.fit(inputs, target)

    def predict(self, data):
        scores = self.model.predict_proba(data.drop(columns=['match']))
        scores = [s[1] for s in scores]
        outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
        return outputs, scores

    def render_roc(self, true_labels, pred_labels_list, threshs, print_stats):
        plot_roc(true_labels, pred_labels_list, threshs, print_stats)

    def print_feature_importances(self):
        pass


class LinearRegressionClassifier(Classifier):
    def __init__(self, weights):
        self.weights = weights
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
        self.weights = weights
        self.model = LogisticRegression()
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        print(f'Feature importances for {self.name} \n {self.model.coef_}')


class SvmClassifier(Classifier):
    def __init__(self, weights):
        self.weights = weights
        self.kernel = 'rbf'  # linear, rbf, poly
        self.model = svm.SVC(kernel=self.kernel, probability=True)
        self.name = str(type(self.model)).split(".")[-1][:-2]

    def print_feature_importances(self):
        if self.kernel == 'linear':
            print(f'Feature importances for {self.name} \n {self.model.coef_}')
        if self.kernel == 'poly' or self.kernel == 'rbf':
            print(f'Support vectors for {self.name} \n {self.model.support_vectors_}')

class NeuralNetworkClassifier(Classifier):
    def __init__(self, weights):
        self.weights = weights
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
        self.weights = weights
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
        self.weights = weights
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
