from math import log2
import operator
import joblib


class Node:

    def __init__(self, parent=None):
        self.parent = parent
        self.children = dict()
        self.selected_feature = None
        self.entropy = None
        self.information_gain = None
        self.label = None


class DecisionTree:

    def __init__(self):
        pass

    @staticmethod
    def entropy_and_label_frequency_calculator(data_labels):
        existing_labels_frequencies = {}
        # counting frequency of each label
        existing_labels_frequencies = data_labels.value_counts().to_dict()
        size = len(data_labels)
        entropy = 0
        for label, f in existing_labels_frequencies.items:
            p = f / size
            entropy += -p * (log2(p) if p else 0)
        return entropy, existing_labels_frequencies

    @staticmethod
    def information_gain_calculator(parent_entropy, childrens_entropies):
        return parent_entropy - sum(childrens_entropies)

    def train(self, X, labels):
        def generate(node, node_data, node_labels):
            node.entropy, label_frequencies = self.entropy_and_label_frequency_calculator(node_labels)
            if not node.entropy:
                # end
                node.label = label_frequencies.keys[0]
                return
                pass
            if node_data.Tables[0].Columns.Count:
                # end
                node.label = max(label_frequencies.iteritems(), key=operator.itemgetter(1))[0]
                return
            # TODO: for each feature assume n children and calculate entropy to calculate IG
            for feature in node_data.columns:
                feature_data = node_data[feature]
                unique_values = feature_data.unique()
                for value in unique_values:
                    indecies = feature_data[feature_data == value].index[0]
                    print(y_train.take(indices=indices, axis=0))

            # TODO: check more than one best IG to choose randomly
            best_IG, best_feature = '', ''
            childrens_entropies = []
            IG = self.information_gain_calculator(self.root.entropy, childrens_entropies)

            node.information_gain = best_IG
            node.selected_feature = best_feature
            for value in feature_values:
                child = Node(parent=node)
                # TODO: get data with this value for the feature and delete feature
                generate(child, child_data, child_labels)
                node.children[value] = child

        # labels = pd.Series(labels)
        self.root = Node()
        generate(self.root, X, labels)

    def predict(self, X):
        pass


classifier = DecisionTree()
# save model
joblib.dump(classifier, 'model.pkl')
# load model
classifier_from_joblib = joblib.load('model.pkl')
# print(type(classifier_from_joblib))

