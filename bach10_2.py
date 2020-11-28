import pandas as pd
import sklearn.naive_bayes as sk_naive_bayes
from probability_plotting import plot_model
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics

# naive bayes

qualifies_double_grade = pd.read_csv('data/double_grade_reevaluated.csv')

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

n_b_model = sk_naive_bayes.GaussianNB()
n_b_model.fit(X, y)

modeled_qualification = n_b_model.predict(X)
confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)
print(confusion_matrix)

plot_model(n_b_model, qualifies_double_grade)

X_probabilities = n_b_model.predict_proba(X)[:, 1]
X_probabilities_log = n_b_model.predict_log_proba(X)[:, 1]

qualifies_double_grade["probability"] = X_probabilities
qualifies_double_grade["log_probabilty"] = X_probabilities_log

pd.set_option("display.max_rows", None)
print(qualifies_double_grade.sort_values(by="probability"))

plt.show()
