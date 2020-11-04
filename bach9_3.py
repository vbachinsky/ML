import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm
import sklearn.metrics as sk_metrics
import numpy as np
import sklearn.model_selection as sk_model_selection


qualifies_double_grade = pd.read_csv('data/double_grade.csv')
regularization_parameter = [0.1, 1, 100]

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

fig, axs = plt.subplots(len(regularization_parameter), 1)

for i in range(len(regularization_parameter)):
    qualification_model = sk_svm.SVC(C=regularization_parameter[i], kernel="linear")
    qualification_model.fit(X, y)
    modeled_qualification = qualification_model.predict(X)
    confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)

    # model evaluation
    number_of_folds = 4
    cv_qualification_model = sk_svm.SVC(C=regularization_parameter[i], kernel="linear")
    cv_model_quality = sk_model_selection.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds,
                                                          scoring="accuracy")

    # plotting
    axs[i].set_xlabel("Technical grade\nRegularization parameter C = {}".format(regularization_parameter[i]))
    axs[i].set_ylabel("English grade")
    qualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 0]
    axs[i].scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
    axs[i].scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")
    xlim = axs[i].get_xlim()
    ylim = axs[i].get_ylim()
    # create grid to evaluate model
    plotting_step = 100
    xx = np.linspace(xlim[0], xlim[1], plotting_step)
    yy = np.linspace(ylim[0], ylim[1], plotting_step)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = qualification_model.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    axs[i].contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5,
                  linestyles=["--", "-", "--"])
    # plot support vectors
    axs[i].scatter(qualification_model.support_vectors_[:, 0], qualification_model.support_vectors_[:, 1],
                  s=200, linewidth=1, facecolors='none', edgecolors="k")

    # output of results
    print("\nFor regularization parameter C = {}".format(regularization_parameter[i]))
    print(cv_model_quality)
    print(confusion_matrix)
    print(qualification_model.coef_)
    print(qualification_model.intercept_)

fig.tight_layout()
plt.show()
