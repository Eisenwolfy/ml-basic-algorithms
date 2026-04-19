import pandas as pd
from sklearn.datasets import load_wine
import math
import numpy as np
from collections import deque

data = load_wine()
x = data.data
y = data.target

class Node:
    def __init__(self, feature=None, children=None, prediction=None, depth=0):
        self.feature = feature
        self.children = children if children is not None else {}
        self.prediction = prediction
        self.depth = depth


class DecisionTreeClassifierScratch:
    def __init__(
            self,
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.target_name_ = None
        self._global_majority_ = None

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        return {
            'criterion' : self.criterion,
            'max_depth' : self.max_depth,
            'min_samples_split' : self.min_samples_split,
            'min_samples_leaf' : self.min_samples_leaf,
            'random_state' : self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            match key:
                case "criterion": self.criterion = value
                case "max_depth": self.max_depth = value
                case "min_samples_split": self.min_samples_split = value
                case "min_samples_leaf": self.min_samples_leaf = value
                case "random_state": self.random_state = value

    # ------------------------------------------------------------------
    # Impurity measures
    # ------------------------------------------------------------------

    def _count_classes(self, y):
        freq = {}
        for i in y:
            freq[i] = freq.get(i, 0) + 1
        return freq

    def _entropy(self, y):
        freq = self._count_classes(y)
        total = len(y)
        res = 0
        for count in freq.values():
            p = count / total
            if p > 0:
                res += p * math.log2(p)
        return -res

    def _gini(self, y):
        freq = self._count_classes(y)
        total = len(y)
        summ = 0
        for count in freq.values():
            p = count/total
            summ += p**2
        res = 1 - summ
        return res

    def _impurity(self, y):
        if self.criterion == "entropy":
            return self._entropy(y)
        elif self.criterion == "gini":
            return self._gini(y)
        else:
            raise ValueError("Unknown criterion")

    # ------------------------------------------------------------------
    # Splitting logic
    # ------------------------------------------------------------------

    def _information_gain(self, data, feature, target):
        parent_entropy = self._entropy(target)

        #next step is to split data
        val = data[:, feature]
        mean = val.mean()
        target_left = target[val <= mean]
        target_right = target[val > mean]

        #correction to avoid incorrect splitting
        if len(target_left) == 0 or len(target_right) == 0:
            return 0
        n = len(target)

        left_weight = len(target_left) / n
        right_weight = len(target_right) / n

        children_entropy = left_weight * self._entropy(target_left) + right_weight * self._entropy(target_right)
        return parent_entropy - children_entropy

    def _information_gain_from_split(self, parent, left, right):
        n = len(parent)

        parent_impurity = self._impurity(parent)

        left_weight = len(left) / n
        right_weight = len(right) / n

        children_impurity = (left_weight * self._impurity(left) + right_weight * self._impurity(right))
        return parent_impurity - children_impurity


    def _best_split(self, data, target):
        #Optimized version of the method (the complexity is )
        best_gain = -1
        best_feat, best_thresh = None, None
        n_samples, n_features = data.shape

        for feat_idx in range(n_features):
            # Sort data by this feature
            thresholds = np.unique(data[:, feat_idx])
            # Instead of every midpoint, just check unique values
            for i in range(len(thresholds) - 1):
                thresh = (thresholds[i] + thresholds[i + 1]) / 2

                left_idx = data[:, feat_idx] <= thresh
                y_left, y_right = target[left_idx], target[~left_idx]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = self._information_gain_from_split(target, y_left, y_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, thresh

        return best_feat, best_thresh, best_gain

    # ------------------------------------------------------------------
    # Majority class — uses np.unique for robustness across array types
    # ------------------------------------------------------------------

    def _majority_class(self, y):
        """
        Returns the most frequent class in y
        Accepts any array-like (list, np.ndarray, pd.Series)
        Uses random_state for reproducible tie-breaking
        """

        freq = self._count_classes(y)

        max_count = max(freq.values())
        candidates = [k for k, v in freq.items() if v == max_count]

        #tie - breaking
        if len(candidates) == 1:
            return candidates[0]

        if self.random_state is not None:
            import random
            random.seed(self.random_state)
            return random.choice(candidates)

        return candidates[0]

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, data, features, target, depth=0):
        node = Node(depth=depth)
        # Leaf conditions
        # 1. Pure node (all samples belong to the same class)
        if len(set(target)) == 1:
            node.prediction = target[0]
            return node

        # 2. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            node.prediction = self._majority_class(target)
            return node

        # 3. Too few samples to split
        if len(target) < self.min_samples_split:
            node.prediction = self._majority_class(target)
            return node

        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(data, target)

        # No useful split found (gain is 0 or all thresholds produced empty sides)
        if best_feature is None or best_gain <= 0:
            node.prediction = self._majority_class(target)
            return node

        # Splitting data again....
        values = data[:, best_feature]
        left_mask = values <= best_threshold
        right_mask = values > best_threshold

        # Enforce min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.prediction = self._majority_class(target)
            return node

        # Build internal node
        node.feature = best_feature
        node.threshold = best_threshold  # store threshold on the node for prediction

        node.children['left']  = self._build_tree(data[left_mask],  features, target[left_mask],  depth + 1)
        node.children['right'] = self._build_tree(data[right_mask], features, target[right_mask], depth + 1)

        return node

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # ROOT FIX: reset both X and y to a clean 0-based index before
        # combining into a single DataFrame, so pandas does not
        # produce NaN rows due to index misalignment (e.g. after
        # train_test_split which preserves the original shuffled index).

        X = X.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True).values
        else:
            y = np.array(y)

        self.classes_ = np.unique(y)
        self.feature_names_in_ = list(X.columns)
        self._global_majority_ = self._majority_class(y)

        data_np = X.values
        features = list(range(data_np.shape[1]))

        self.tree_ = self._build_tree(data_np, features, y, depth=0)
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _collect_all_leaf_labels(self, node):
        """Recursively collects ALL leaf labels — ensures unbiased majority vote."""
        if node.prediction is not None:
            return [node.prediction]
        labels = []
        for child in node.children.values():
            labels.extend(self._collect_all_leaf_labels(child))
        return labels

    def _predict_one(self, row, node):
        # Leaf node — return its prediction
        if node.prediction is not None:
            return node.prediction

        # Should not happen in a well-built tree, but as a guard
        if node.feature is None or not node.children:
            labels = self._collect_all_leaf_labels(node)
            return self._majority_class(labels) if labels else self._global_majority_

        # Traverse left or right based on threshold
        if row[node.feature] <= node.threshold:
            child = node.children.get('left')
        else:
            child = node.children.get('right')

        # Missing branch fallback
        if child is None:
            labels = self._collect_all_leaf_labels(node)
            return self._majority_class(labels) if labels else self._global_majority_

        return self._predict_one(row, child)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.reset_index(drop=True)
        data_np = X.values
        return np.array([self._predict_one(row, self.tree_) for row in data_np])

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def score(self, X, y):
        predictions = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)
        return np.mean(predictions == y)

    # ------------------------------------------------------------------
    # Utility: pretty-print tree structure (no Graphviz)
    # ------------------------------------------------------------------
    """
    Since I'm not using Graphviz, I decided to use sklearn instead
    To do it i need to implement 2 additional methods (to_slearn_tree, plot)
    The first one converts this scratch into a sklearn classifier with our node structure
    This allows us to use all skearn's visualization tools without Graphviz
    """
    def to_sklearn_tree(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree._tree import Tree

        if self.tree_ is None:
            raise RuntimeError("Call fit() before to_sklearn_tree().")

        X = np.array(X) if not isinstance(X, np.ndarray) else X   #the training data — used to recount samples per node.
        y = np.array(y) if not isinstance(y, np.ndarray) else y   #the training labels.

        classes = self.classes_
        n_classes = len(classes)
        n_features = X.shape[1]

        # Breadth First Search to assign stable node IDs
        nodes_ordered = []
        queue = deque([self.tree_])
        node_to_id = {}
        i = 0
        while queue:
            node = queue.popleft()
            node_to_id[id(node)] = i
            nodes_ordered.append(node)
            i += 1
            if "left"  in node.children: queue.append(node.children["left"])
            if "right" in node.children: queue.append(node.children["right"])

        n = len(nodes_ordered)

        # Allocate arrays
        children_left = np.full(n, -1, dtype=np.intp)
        children_right = np.full(n, -1, dtype=np.intp)
        feature = np.full(n, -2, dtype=np.intp) # -2 = leaf sentinel
        threshold = np.full(n, -2.0, dtype=np.float64)
        impurity = np.zeros(n, dtype=np.float64)
        n_node_samples = np.zeros(n, dtype=np.intp)
        # value must hold CLASS PROPORTIONS (sum = 1 per node) for filled=True colors
        value = np.zeros((n, 1, n_classes), dtype=np.float64)

        # Propagate sample indices through the tree
        sample_indices = [None] * n
        sample_indices[0] = np.arange(len(y))

        for idx, node in enumerate(nodes_ordered):
            idxs = sample_indices[idx]
            if idxs is None:
                idxs = np.array([], dtype=int)

            n_node_samples[idx] = len(idxs)

            counts = np.array(
                [np.sum(y[idxs] == cls) for cls in classes], dtype=np.float64
            )
            total = counts.sum()
            value[idx, 0, :] = counts / total if total > 0 else counts

            if total > 0:
                probs = counts / total
                probs = probs[probs > 0]
                impurity[idx] = -np.sum(probs * np.log2(probs))   # entropy

            if "left" in node.children:
                lid = node_to_id[id(node.children["left"])]
                rid = node_to_id[id(node.children["right"])]
                children_left[idx] = lid
                children_right[idx] = rid
                feature[idx] = node.feature
                threshold[idx] = node.threshold

                col  = X[:, node.feature]
                mask = col[idxs] <= node.threshold
                sample_indices[lid] = idxs[mask]
                sample_indices[rid] = idxs[~mask]

        # Build sklearn Tree object via __setstate__
        t = Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
        weighted = n_node_samples.astype(np.float64)
        t.__setstate__({
            "max_depth":  int(self.max_depth) if self.max_depth is not None else int(np.max([nd.depth for nd in nodes_ordered])),
            "node_count": n,
            "nodes": np.array(
                list(zip(children_left, children_right, feature, threshold,
                         impurity, n_node_samples, weighted, [True] * n)),
                dtype=[
                    ("left_child", "<i8"),
                    ("right_child", "<i8"),
                    ("feature", "<i8"),
                    ("threshold", "<f8"),
                    ("impurity", "<f8"),
                    ("n_node_samples", "<i8"),
                    ("weighted_n_node_samples", "<f8"),
                    ("missing_go_to_left", "u1"),
                ],
            ),
            "values": value,
        })

        # Wrap in a proxy sklearn classifier, (a fitted-looking sklearn tree can be passed directly to plot_tree)
        proxy = DecisionTreeClassifier()
        proxy.tree_ = t
        proxy.n_features_in_ = n_features
        proxy.classes_ = classes
        proxy.n_classes_ = n_classes
        proxy.n_outputs_ = 1
        proxy.max_features_ = n_features
        return proxy

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, X, y, feature_names=None, class_names=None, figsize=(20, 10), fontsize=10, savepath=None):
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        proxy = self.to_sklearn_tree(X, y)
        feature_names = feature_names or self.feature_names_in_
        if class_names is None and self.classes_ is not None:
            class_names = [str(c) for c in self.classes_]

        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            proxy,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=fontsize,
        )
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"Tree plot saved to: {savepath}")

        plt.show()
        return fig, ax


# And now we need to test our class
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    my_tree = DecisionTreeClassifierScratch()
    wine = load_wine()
    Xw, yw = load_wine(return_X_y=True)
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42)
    my_tree.fit(Xw_train, yw_train)
    train_w = my_tree.score(Xw_train, yw_train)
    test_x = my_tree.score(Xw_test, yw_test)
    print(f'Train accuracy: {train_w:.4f}')
    print(f'Test accuracy: {test_x:.4f}')


    my_tree.plot(
        X = x,
        y = y,
        feature_names = list(wine.feature_names),
        class_names=list(wine.target_names),
        figsize=(24, 10),
        fontsize=10,
        savepath="tree_plot.png"
    )