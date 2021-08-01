from score_computation.images_and_names.compute_total_similarity import plot_roc
from sklearn.linear_model import LinearRegression

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
        outputs = [ 0 if score < self.weights['threshold'] else 1 for score in scores ]
        return outputs

    def render_roc(self, true_labels, pred_labels_list, threshs, print_stats):
        plot_roc(true_labels, pred_labels_list, threshs, print_stats)
