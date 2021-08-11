from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RandomForests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from evaluate_classifier import plot_roc


class LinearClassifier:

    def __init__(self, weights):
        self.weights = weights
        self.model = LinearRegression()

    def fit(self, data):
        target = data['match']
        inputs = data.drop(columns=['match'])
        self.model.fit(inputs, target)

    def predict(self, data):
        scores = self.model.predict(data.drop(columns=['match']))
        outputs = [0 if score < self.weights['threshold'] else 1 for score in scores]
        return outputs, scores

    def render_roc(self, true_labels, pred_labels_list, threshs, print_stats):
        plot_roc(true_labels, pred_labels_list, threshs, print_stats)


class LogisticRegressionClassifier:

    def __init__(self, weights):
        self.weights = weights
        self.model = LogisticRegression()

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


class SvmClassifier:

    def __init__(self, weights):
        self.weights = weights
        self.model = svm.SVC(kernel='poly', probability=True)  # linear, rbf, sigmoid, poly

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


class NeuralNetworkClassifier:
    def __init__(self, weights):
        self.weights = weights
        self.model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', solver='adam', max_iter=1000)

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


class DecisionTreeClassifier:
    def __init__(self, weights):
        self.weights = weights
        self.model = DecisionTree(max_depth=10)

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


class RandomForestsClassifier:
    def __init__(self, weights):
        self.weights = weights
        self.model = RandomForests(max_depth=10, n_estimators=10)

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
