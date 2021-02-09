import pandas as pd
import sklearn.preprocessing as sk_preprocessing
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.pipeline as sk_pipeline
import sklearn.svm as sk_svm
import sklearn.ensemble as sk_ensemble
import sklearn.linear_model as sk_linear
import sklearn.linear_model as sk_linear_models


diabetes_df = pd.read_csv("data/pima-indians-diabetes.csv").sample(frac=1)
column_names = diabetes_df.columns.values

X = diabetes_df[column_names[:-1]]
y = diabetes_df[column_names[-1]]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)

k_folds = 4

# Bagging with SVM
bagging_svm_model = sk_pipeline.Pipeline([("scaler", sk_preprocessing.StandardScaler()),
                                          ("model",
                                           sk_ensemble.BaggingClassifier(base_estimator=sk_svm.SVC(kernel="rbf",
                                                                                                   probability=True),
                                                                         n_jobs=-1))])
bagging_svm_result = sk_model_selection.cross_val_score(bagging_svm_model, X_train, y_train, cv=k_folds)
print("Bagging with SVC accuracy: {:.2f}%".format(bagging_svm_result.mean()*100))

# Random Forest
rfc_model = sk_ensemble.RandomForestClassifier(n_jobs=-1)
rfc_result = sk_model_selection.cross_val_score(rfc_model, X_train, y_train, cv=k_folds)
print("RFC accuracy: {:.2f}%".format(rfc_result.mean()*100))

# Logistic Regression
l_r_model = sk_linear_models.LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, C=0.01)
l_r_result = sk_model_selection.cross_val_score(l_r_model, X_train, y_train, cv=k_folds)
print("LR accuracy: {:.2f}%".format(l_r_result.mean()*100))

# Ensemble
estimators = []
estimators.append(("Bagging_SVM", bagging_svm_model))
estimators.append(("LR", l_r_model))
estimators.append(("RFC", rfc_model))

meta_estimator = sk_linear.LogisticRegression()

ensemble_model = sk_ensemble.StackingClassifier(estimators=estimators, final_estimator=meta_estimator)
ensemble_model.fit(X_train, y_train)
ensemble_prediction = ensemble_model.predict(X_test)

print("\nStacking accuracy: {:.2f}%".format(sk_metrics.accuracy_score(y_test, ensemble_prediction)*100))
