class KNearestNeighbors:
    """
    K-Nearest-Neighbors classifier
    """
    def __init__(self, k: int):
        """
        Initialize function
        :param k: k value
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.feature_count = 0
        self.data_size = 0

    def fit(self, X_train, y_train):
        """
        Determining train data and their labels plus number of columns and rows
        :param X_train: train data
        :param y_train: train data labels
        :return: None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.feature_count = len(X_train.columns)
        self.data_size = len(X_train)

    def euclidean_distance(self, x1, x2):
        """
        Calculating euclidean distance of two data
        :param x1: first datum
        :param x2: second datum
        :return: distance
        """
        s = 0
        for i in range(self.feature_count):
            s += (x1[i] - x2[i]) ** 2
        return s ** 0.5

    def predict(self, X_test):
        """
        Predicting labels of test data by KNN algortihm
        :param X_test: test data
        :return: test data predicted labels
        """

        #  Constraint for ensuring calling fit function
        if not self.data_size:
            raise Exception('Error: Function fit() must be called before predict!')

        predictions = []
        for i, x1 in X_test.iterrows():
            neighbors = []
            # Calculating distance from test datum x1 to all train data
            for j, x2 in self.X_train.iterrows():
                neighbors.append((self.euclidean_distance(x1, x2), j))
            # Sorting distances to find k-nearest
            neighbors.sort()
            # Finding label based on the number of Won and Lost labels of k-nearest
            score = 0
            for g in range(self.k):
                score += 1 if self.y_train[neighbors[g][1]] == 'Won' else -1
            if score > 0:
                predictions.append('Won')
            else:
                predictions.append('Lost')
        return predictions
