class LinearClassifier:

    def __init__(self, weights):
        self.weights = weights

    def fit(self, data):
        pass

    def predict(self, inputs):
        outputs = []
        for index, pair_similarities in inputs.iterrows():
            score = 0
            for attribute in pair_similarities.keys():
                if attribute in self.weights:
                    score += self.weights[attribute] * pair_similarities[attribute]

            outputs.append(0 if score < self.weights['threshold'] else 1)

        return outputs

    def render_roc(self):
        pass