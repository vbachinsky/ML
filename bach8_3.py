import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear_model
import sklearn.metrics as sk_metrics


def train_linear_model(X, y):
    linear_regression = sk_linear_model.LinearRegression()
    linear_regression.fit(X, y)
    return linear_regression


def train_logistic_model(X, y):
    logical_regression = sk_linear_model.LogisticRegression()
    logical_regression.fit(X, y)
    return logical_regression


def predictor(x):
    return x * linear_qualification_model.coef_[0] + linear_qualification_model.intercept_


qualifies_single_grade = pd.read_csv('data/linear_vs_logistic.csv')
qualifies_single_grade.sort_values(by=["grade", "qualifies"], inplace=True)

X = qualifies_single_grade[["grade"]]
y = qualifies_single_grade["qualifies"]

linear_qualification_model = train_linear_model(X, y)
logistic_qualification_model = train_logistic_model(X, y)

# Predict
linear_modeled_qualification = linear_qualification_model.predict(X)
logistic_modeled_qualification = logistic_qualification_model.predict(X)

# Probability
logistic_modeled_qualification_probability = logistic_qualification_model.predict_proba(X)[:, 1]
qualifies_single_grade["logistic_modeled_probability"] = logistic_modeled_qualification_probability
qualifies_single_grade["linear_modeled_probability"] = [0 if predictor(i) < 0 else 1 if predictor(i) > 1 else predictor(i) for i in qualifies_single_grade["grade"]]

print(qualifies_single_grade)

# Calculate confusion matrix
logistic_confusion_matrix = sk_metrics.confusion_matrix(y, logistic_modeled_qualification)
print("Confusion matrix for logistic model: \n", logistic_confusion_matrix)

all_probability_negative = [0 if i < 0.5 else 1 for i in qualifies_single_grade["linear_modeled_probability"]].count(0)
all_negative = [0 if i == 0 else 1 for i in qualifies_single_grade["qualifies"]].count(0)
true_probability_negative = 0
for z in range(len(qualifies_single_grade)):
    if qualifies_single_grade.at[z, "qualifies"] ==0 and qualifies_single_grade.at[z, "linear_modeled_probability"] <= 0.5:
        true_probability_negative +=1
print(all_probability_negative, all_negative, true_probability_negative)

all_probability_positive = [1 if i >= 0.5 else 0 for i in qualifies_single_grade["linear_modeled_probability"]].count(1)
all_positive = [1 if i == 1 else 0 for i in qualifies_single_grade["qualifies"]].count(1)
true_probability_positive = 0
for z in range(len(qualifies_single_grade)):
    if qualifies_single_grade.at[z, "qualifies"] == 1 and qualifies_single_grade.at[z, "linear_modeled_probability"] > 0.5:
        true_probability_positive +=1
print(all_probability_positive, all_positive, true_probability_positive)
conf_matrix_linear = [[true_probability_negative, all_negative - true_probability_negative],
                      [all_positive - true_probability_positive, true_probability_positive]]
print("Confusion matrix for linear model: \n", conf_matrix_linear)

# Accuracy, error, precision, recall
print("\nFor logistic model:")
print(f"Accuracy: {sk_metrics.accuracy_score(y, logistic_modeled_qualification)}")
print(f"Error: {1-sk_metrics.accuracy_score(y, logistic_modeled_qualification)}")
print(f"Precision: {sk_metrics.precision_score(y, logistic_modeled_qualification)}")
print(f"Recall: {sk_metrics.recall_score(y, logistic_modeled_qualification)}")

print("\nFor linear model:")
print(f"Accuracy: {(true_probability_positive + true_probability_negative)/len(qualifies_single_grade)}")
print(f"Error: {1 - ((true_probability_positive + true_probability_negative)/len(qualifies_single_grade))}")
print(f"Precision: {true_probability_positive/all_probability_positive}")
print(f"Recall: {true_probability_positive/all_positive}")

# Plotting
fig, axs = plt.subplots(2, 1)
axs[0].scatter(X, y)
axs[0].plot(X, linear_modeled_qualification, color='r')
axs[0].plot(X, logistic_modeled_qualification, color='g')
axs[0].set_xlabel('grade')
axs[0].set_ylabel('qualifies')

axs[1].scatter(X, y)
axs[1].plot(X, logistic_modeled_qualification_probability, color='g')
axs[1].plot(X, qualifies_single_grade["linear_modeled_probability"])
axs[1].set_xlabel('grade')
axs[1].set_ylabel('probability')

fig.tight_layout()
plt.show()

qualifies_single_grade.to_csv(("data/prepared_data_linear_vs_logistic.csv"))
