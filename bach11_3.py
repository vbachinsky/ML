import pandas as pd
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble

# tree and forest

def convert_to_numeric_values(df):
    converted_df = df.copy()
    converted_df = converted_df.replace({"class": {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}})
    return converted_df


iris_df = pd.read_csv("data/iris.csv")
numeric_iris_df = convert_to_numeric_values(iris_df)
print(numeric_iris_df)

column_names = iris_df.columns.values

X = numeric_iris_df[column_names[:-1]]
y = numeric_iris_df[column_names[-1]]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y)

# Decision tree
print("\nTree")
max_depth = 5
for z in range(1, max_depth + 1):
    tree_classifier = sk_tree.DecisionTreeClassifier(max_depth=z)
    tree_classifier.fit(X_train, y_train)
    tree_y_prediction = tree_classifier.predict(X_test)

    print("The maximum depth of the tree: {}. Accuracy: {}".format(z, sk_metrics.accuracy_score(y_test, tree_y_prediction)))

    tree_confusion_matrix = sk_metrics.confusion_matrix(y_test, tree_y_prediction)
    print(tree_confusion_matrix)

# Random forest
print("\nForest")
max_estimators = 21
for z in range(10, max_estimators, 2):
    tree_classifier = sk_ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=z)
    tree_classifier.fit(X_train, y_train)
    tree_y_prediction = tree_classifier.predict(X_test)

    print("The number of trees in the forest: {}. Accuracy: {}".format(z, sk_metrics.accuracy_score(y_test, tree_y_prediction)))

    tree_confusion_matrix = sk_metrics.confusion_matrix(y_test, tree_y_prediction)
    print(tree_confusion_matrix)
