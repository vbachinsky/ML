import pandas as pd
import sklearn.neighbors as sk_neighbours
import numpy as np
import sklearn.model_selection as sk_model_selection
import matplotlib.pyplot as plt


# метод k-ближайших соседей

def plot_model(model, qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])

    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    plt.contourf(probability_matrix, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv('data/double_grade_reevaluated.csv')

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

for k in range(1, 10, 2):
    print(f"{k} neighborurs:")
    double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=k)
    cv_double_grade_model_quality = sk_model_selection.cross_val_score(double_grade_knn_model, X, y, cv=4,
                                                                       scoring="accuracy")
    print("Accuracy: {}".format(np.mean(cv_double_grade_model_quality)))


# 1 neighborurs:
# Accuracy: 0.8999999999999999
# 3 neighborurs:
# Accuracy: 0.9299999999999999
# 5 neighborurs:
# Accuracy: 0.9099999999999999
# 7 neighborurs:
# Accuracy: 0.89


double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=3)
double_grade_knn_model.fit(X, y)

plot_model(double_grade_knn_model, qualifies_double_grade_df)

plt.show()
