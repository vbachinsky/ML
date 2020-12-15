import sklearn.neural_network as sk_n_n
import pandas as pd
import sklearn.preprocessing as sk_preprocessing
import sklearn.model_selection as sk_model_selection
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics


qualifies_double_grade_df = pd.read_csv('data/double_grade_reevaluated.csv')

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

X = sk_preprocessing.StandardScaler(with_mean=False).fit_transform(X)
y = sk_preprocessing.LabelEncoder().fit_transform(y)

fig, axs = plt.subplots(4, 1)
model_conditions = [[10, 0.001], [15, 0.0001], [20, 0.00001], [40, 0.000001]]

for i in range(len(model_conditions)):
    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y)

    qualification_model = sk_n_n.MLPClassifier(max_iter=2000, verbose=False,
                                               hidden_layer_sizes=(model_conditions[i][0],),
                                               alpha=model_conditions[i][1]).fit(X_train, y_train)
    print("For {} neurons in hidden layer and alpha = {} accuracy = {}".format(model_conditions[i][0],
                                                                               model_conditions[i][1],
                                                                               qualification_model.score(X_test,
                                                                                                         y_test)))
    modeled_qualification = qualification_model.predict(X)
    confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)
    print(confusion_matrix)

    axs[i].set_xlabel("Technical grade. For {} neurons in hidden layer and alpha = {}".format(model_conditions[i][0],
                                                                                              model_conditions[i][1]))
    axs[i].set_ylabel("English grade")

    qualified_candidates = X[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = X[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 8
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])

    probability_levels = qualification_model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    axs[i].contourf(probability_matrix, cmap="rainbow")

    axs[i].scatter(qualified_candidates[:, 0], qualified_candidates[:, 1], color="w")
    axs[i].scatter(unqualified_candidates[:, 0], unqualified_candidates[:, 1], color="k")

fig.tight_layout()
plt.show()
