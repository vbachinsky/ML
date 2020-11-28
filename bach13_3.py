import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.model_selection as sk_model_selection


diabetes_df = pd.read_csv("data/pima-indians-diabetes.csv")

X = diabetes_df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                 "DiabetesPedigreeFunction", "Age"]]
y = diabetes_df["Class"]

standard_scaler = sk_preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

for n_components in reversed(range(1, 8)):
    principal_components = sk_decomposition.PCA(n_components=n_components).fit_transform(X)
    cv_diabetes_log_model = sk_linear.LogisticRegression()
    cv_diabetes_log_model_quality = sk_model_selection.cross_val_score(cv_diabetes_log_model, principal_components,
                                                                       y, cv=4, scoring="accuracy")
    print(f"\nFor {n_components} components quality: ")
    print(np.mean(cv_diabetes_log_model_quality))

pca_all_components = sk_decomposition.PCA()
pca_all_components.fit(X)

print("\nExplained variance ratio:")
print(pca_all_components.explained_variance_ratio_)

components = list(range(1, pca_all_components.n_components_ + 1))
plt.plot(components, np.cumsum(pca_all_components.explained_variance_ratio_), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.ylim(0, 1.1)

plt.show()
