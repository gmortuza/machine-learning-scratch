# Necessary libraries
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing
import numpy as np
import random


class Node:
    """
    Represents a single tree node
    """
    def __init__(self, regions, loss=None, index=None, threshold=None, left=None, right=None):
        """

        :param regions: training data point of this region
        :param loss: contains the loss of this node. This is the measure of impurity. loss 0 means
                     this node contains only 1 class
        :param index: Index value of the data point that is used as a threshold
        :param threshold: threshold value of the point that is used to split the data
        :param left: left child of this node
        :param right: right child of this node
        """
        self.regions = regions
        self.loss = loss
        self.index = index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = False
        self.output = None

    def make_leaf(self):
        """
        Make this node as a leaf node of the tree
        :return:
        """
        self.is_leaf = True
        # The majority of the output class in this node's region's dataset
        self.output = self.__calculate_output()
        # Leaf node will not have any children
        self.left = self.right = None

    def __calculate_output(self):
        """
        Resulting class of this node.
        :return:
        """
        # Extract the label only from the regions
        outputs = self.regions[:, -1]
        # Get the number of occurance of each class
        values, counts = np.unique(outputs, return_counts=True)
        # return the highest occured class
        return int(values[np.argmax(counts)])

    def __len__(self):
        """
        length of the class will be the length of the region's datapoint
        :return:
        """
        return self.regions.shape[0]

    def __str__(self):
        # TODO: Make a str
        pass


class DecisionTree:
    """
    A non-linear, non-parametric supervised machine learning algorithm that is used for classification purpose
    It uses a greedy, top-down, recursive partitioning system to get the output result.
    In this approach we have implemented this tree using a Depth First Search (DFS)
    Run time of this algorithm is:
        Training --> O(m * num_features * depth_of_tree)
        Tesing --> O(depth_of_tree)
    """
    def __init__(self, loss_function="gini", maximum_depth=100, min_leaf_size=4):
        """
        The parameter of this method are used to regularize the decision tree model.
        Decision Tree model have high tendency to get overfit.
        There are couple more regularizing parameter. i.e. maximum number of node in the overall tree, pruning etc.
        :param maximum_depth: Maximum depth of the tree
        :param min_leaf_size: Minimum number of data in any individual nodes
        :param loss_function: Loss function that will be used. gini/entropy/misclass
        :return:
        """
        # Number of output class
        self.num_class = None
        # Number of training example
        self.m = None
        # Number of input feature
        self.n_features = None
        # root node of our tree
        self.root = None
        # Loss function
        self.loss_func = loss_function
        # regularization parameter
        self.max_depth = maximum_depth
        self.min_leaf_size = min_leaf_size

    def fit(self, X_train, y_train):
        """
        Train the model
        :param X_train: Training dataset
        :param y_train: Training label
        :return:
        """
        self.num_class = np.unique(y_train).shape[0]
        self.m, self.n_features = X_train.shape
        # Preprocess the data
        X_train = preprocessing.scale(X_train)
        # Expand the dimension of training label to merge it with the test label
        # [1 0 1 0 1 1 1] ==> [[1] [0] [1] [0] [1] [1] [1]]
        y_train = np.expand_dims(y_train, axis=-1)
        # The last column of the region will contain the label for any individual datapoint
        # So that we don't need to pass the labels seperately
        region = np.concatenate((X_train, y_train), axis=1)
        # The whole dataset is our root region
        self.root = Node(region)
        # Split root in left and right region
        self.__split(self.root, self.loss_func)
        # Analyze left and right regions. Either split those regions or make those regions as leaf
        self.__build_tree(self.root, self.loss_func, self.max_depth, self.min_leaf_size, 1)

    def predict(self, X_test):
        """
        Test the model
        :param X_test: Dataset to be tested on
        :return:
        """
        # Scale the testing dataset first
        X_test = preprocessing.scale(X_test)
        result = []
        for single_test in X_test:  # handle each of the test point separately
            result.append(self.__single_predict(single_test, self.root))
        return result

    def __get_loss(self, regions, loss_func):
        """
        Calculate the loss index of any regions
        :param regions: can have multiple regions
        :return:
        """
        total_loss = 0.0
        # Number of data points in all the regions
        training_example = sum([region.shape[0] for region in regions])
        for region in regions:  # Calculate loss index for each of the regions
            # The last row of the regions contains it's label
            # Extract the labels
            label = region[:, -1]
            # If there is no data point then the loss will be 0
            if label.shape[0] == 0:
                continue
            # Gini index for a single region
            individual_region_loss = 0.0
            # Number of appearance of each class in a region
            _, counts = np.unique(label, return_counts=True)
            for count in counts:
                # proportion of that class in that region
                p = count / label.shape[0]
                # See math in the PDF attached
                if loss_func == 'entropy':
                    print("entropy")
                    individual_region_loss -= p * np.log(p)
                else:
                    individual_region_loss += p * (1 - p)
            # Normalize the region's loss and add that to the overall loss index
            total_loss += (label.shape[0] * individual_region_loss) / training_example
        return total_loss

    def _split_by_index(self, index, threshold, region):
        """
        Split any regions based on a threshold value
        :param index: index on which we will apply this threshold. It represents the input feature
        :param threshold: Threshold that will be used for splitting
        :param region: region on which we will do the splitting
        :return: splitted regions
        """
        # data points that have value less than the threshold for a specific input feature(index)
        less_threshold = region[:, index] < threshold
        # Left region will contain the points that didn't have value more then threshold
        left_region = region[less_threshold]
        right_region = region[~less_threshold]
        return left_region, right_region

    def __split(self, node, loss_func):
        """
        Do all possible splitting and choose the best splitting based on the loss function(loss, entropy)
        :param node: instance of class Node. Where it will only have the data points no left/right node. This method
                     will add the left and right node based on the loss function
        :param loss_func: loss function to be used. gini/entropy
        :return:
        """
        # Data points of this node
        region = node.regions
        # Initially the loss is the infinity or comparing purpose
        optimum_loss = float('inf')
        # calculate loss for each data point and each feature
        for row in region:  # iter over the data points
            for index in range(self.n_features):  # iter over the features
                # Get the split
                left, right = self._split_by_index(index, row[index], region)
                # Calculate loss index for that split
                loss = self.__get_loss([left, right], loss_func)
                # if the loss is less than previous calculated loss then update optimum values
                if loss < optimum_loss:
                    optimum_loss = loss
                    optimum_index = index
                    optimum_threshold = row[index]
                    optimum_left = left
                    optimum_right = right
        # Save the optimum value into the nodes
        node.loss = optimum_loss
        node.index = optimum_index
        node.threshold = optimum_threshold
        # If the loss index is zero then it means that this node is pure/homogenous
        # There is only one class in this node
        # So we can make it as leaf node
        if optimum_loss == 0:
            node.make_leaf()
            return
        node.left = Node(optimum_left)
        node.right = Node(optimum_right)

    def __build_tree(self, node, loss_func, max_depth, min_leaf_size, current_depth):
        """
        We will build the tree using a depth first search approach
        :param node: Node that we will be working on
        :param max_depth: maximum depth of the tree
        :param min_leaf_size: minimum leaf size. regularization parameter
        :param current_depth: Current depth of the node
        :return:
        """
        # leaf node need no splitting/processing
        if node.is_leaf or node is None:
            return
        # I either of the node doesn't exists then we will make that as leaf node
        if node.left is None or node.right is None or current_depth > max_depth:
            node.make_leaf()

        # process the left child
        # If if the node has less data point than min_leaf_size then we can make it leaf node
        if len(node.left) < min_leaf_size:
            # make this as leaf
            node.left.make_leaf()
        else:
            # Split it
            # It will assign left and right child based on the optimum loss function
            self.__split(node.left, loss_func)
            # Process the left child of this node which is just splitted
            self.__build_tree(node.left, loss_func, max_depth, min_leaf_size, current_depth+1)

        # Process right child
        if len(node.right) < min_leaf_size:
            node.right.make_leaf()
        else:
            self.__split(node.right, loss_func)
            self.__build_tree(node.right, loss_func, max_depth, min_leaf_size, current_depth+1)

    def __single_predict(self, data, current_node):
        """
        Predict a single data points
        :param data: data to be predicted
        :param current_node:
        :return:
        """
        # If we are in the leaf node already we can send the result
        if current_node.is_leaf:
            return current_node.output
        # The left side of the tree contains the value less than the threshold
        # if data point's value less than threshold then we will process left child
        if data[current_node.index] < current_node.threshold:
            # we will go left
            return self.__single_predict(data, current_node.left)
        else:
            return self.__single_predict(data, current_node.right)


if __name__ == '__main__':
    random.seed(1)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    print(f"F1 score of sklearn model is : {f1_score(y_test, clf.predict(X_test), average='micro')}")
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    print(f"F1 socre of our model is: {f1_score(dt.predict(X_test), y_test, average='micro')}")
