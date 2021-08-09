from sklearn import svm
from sklearn.linear_model import LinearRegression

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


class SvmClassifier:

    def __init__(self, weights):
        self.weights = weights
        self.model = svm.SVC(kernel='linear', probability=True)  # rbf

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
