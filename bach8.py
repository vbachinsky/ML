import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear_models
import sklearn.metrics as sk_metrics


qualifies_single_grade = pd.read_csv('data/single_grade.csv')
qualifies_single_grade.sort_values(by=["grade", "qualifies"], inplace=True) # inplace  заменяет исходный набор на отсортированный
print(qualifies_single_grade)

X = qualifies_single_grade[["grade"]]
y = qualifies_single_grade["qualifies"]

plt.scatter(X, y)

qualificaion_model = sk_linear_models.LogisticRegression()
qualificaion_model.fit(X, y)        # обучение

model_qualification = qualificaion_model.predict(X)
model_qualification_probability = qualificaion_model.predict_proba(X)[:, 1]     # предсказанные данные из модели

qualifies_single_grade["modeled probability"] = model_qualification_probability
print(qualifies_single_grade)

confusion_matrix = sk_metrics.confusion_matrix(y, model_qualification)  # матрица спутанности
print(confusion_matrix)
# [[19  3]
#  [ 2 16]]
# Acc = (19+16)/40 = 0.875
# Err = 1-Acc = (2+3)/40 = 0.125
# Pr = 16/(3+16) = 0.84 - точнсть
# Rec = 16/(2+16) = 0.89 - чувствительность

sk_metrics.accuracy_score(y, model_qualification)
print(f"Accuracy: {sk_metrics.accuracy_score(y, model_qualification)}")
print(f"Error: {1-sk_metrics.accuracy_score(y, model_qualification)}")
print(f"Recall: {sk_metrics.recall_score(y, model_qualification)}")


plt.plot(X, model_qualification, color="r")
plt.plot(X, model_qualification_probability, color="g")
plt.show()

