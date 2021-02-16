import numpy as np
from sklearn.base import BaseEstimator

def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE

    probs = np.mean(y, axis=0)
    res = np.sum(probs * np.log(probs + EPS))

    return -res


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE

    return 1 - np.sum((np.mean(y, axis=0)) ** 2)


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE

    return np.sum((y - np.mean(y)) ** 2) / y.shape[0]


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE

    return np.sum(np.abs(y - np.median(y))) / y.shape[0]


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    Added pred_label by myself
    """

    def __init__(self, feature_index, threshold, pred_label=None, proba=0.):
        self.feature_index = feature_index
        self.value = threshold
        self.pred_label = pred_label  # self-made)
        self.proba = proba
        self.left_child = None
        self.right_child = None
        self.depth = 0


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        for i in range(0, X_subset.shape[0]):
            if X_subset[i][feature_index] < threshold:
                X_left.append(X_subset[i])
                y_left.append(y_subset[i])
            else:
                X_right.append(X_subset[i])
                y_right.append(y_subset[i])

        X_left = np.array(X_left)
        y_left = np.array(y_left)
        X_right = np.array(X_right)
        y_right = np.array(y_right)

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        y_left = []
        y_right = []

        for i in range(0, X_subset.shape[0]):
            if X_subset[i][feature_index] < threshold:
                y_left.append(y_subset[i])
            else:
                y_right.append(y_subset[i])

        y_left = np.array(y_left)
        y_right = np.array(y_right)

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        m = y_subset.size
        if m <= 1:
            return None, None
        criterion = self.criterion
        min_criterion = 100  # Start value
        feature_index = 0
        threshold = 0
        n_features = X_subset.shape[1]
        for i in range(n_features):
            for t in np.unique(X_subset[:, i]):
                y_left, y_right = self.make_split_only_y(i, t, X_subset, y_subset)
                new_criterion = (y_left.shape[0] / y_subset.shape[0]) * criterion(y_left) + \
                                (y_right.shape[0] / y_subset.shape[0]) * criterion(y_right)
                if new_criterion < min_criterion:
                    min_criterion = new_criterion
                    feature_index = i
                    threshold = t
        return feature_index, threshold

    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if self.depth < depth:
            self.depth = depth

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        node = Node(feature_index, threshold, proba=0)
        node.depth = depth
        if depth < self.max_depth and feature_index is not None:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            node.left_child = self.make_tree(X_left, y_left, node.depth + 1)
            node.right_child = self.make_tree(X_right, y_right, node.depth + 1)
        else:
            if self.criterion_name in ['gini', 'entropy']:
                node.proba = np.mean(y_subset, axis=0)
                node.value = np.argmax(node.proba)
            elif self.criterion == "variance":
                node.value = np.mean(y_subset, axis=(0, 1))
            elif self.criterion_name == "mad_median":
                node.value = np.median(y_subset, axis=(0, 1))

        return node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def pred(self, X):
        node = self.root
        while node.depth < self.max_depth:
            if X[node.feature_index] < node.value and node.left_child is not None:
                node = node.left_child
            elif node.right_child is not None:
                node = node.right_child
        return node.value

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        y_predicted = []

        for i in range(0, X.shape[0]):
            y_predicted.append(self.pred(X[i]))

        return np.array(y_predicted)

    def pred_prob(self, X_i):
        node = self.root
        while node.depth < self.max_depth:
            if X_i[node.feature_index] < node.value and node.left_child is not None:
                node = node.left_child
            elif node.right_child is not None:
                node = node.right_child
        return node.proba

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE

        y_predicted_probs = []
        for i in range(0, X.shape[0]):
            y_predicted_probs.append(self.pred_prob(X[i]))

        return np.array(y_predicted_probs)
