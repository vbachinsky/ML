import sklearn.neural_network as sk_nn
import pandas as pd
import sklearn.preprocessing as sk_preprocessing
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.pipeline as sk_pipeline
import sklearn.svm as sk_svm
import sklearn.ensemble as sk_ensemble
import sklearn.linear_model as sk_linear


# Stacking

qualifies_double_grade_df = pd.read_csv('data/double_grade_reevaluated.csv')

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)

k_folds = sk_model_selection.StratifiedKFold(n_splits=4, shuffle=True)  # разбивает на k-сгибов так, что соотношение классов сохраняется

# Neural Network
ann_model = sk_pipeline.Pipeline([("scaler", sk_preprocessing.StandardScaler()),
                                  ("model", sk_nn.MLPClassifier(hidden_layer_sizes=(10, 10),
                                                                activation="tanh", max_iter=100000))])

ann_result = sk_model_selection.cross_val_score(ann_model, X_train, y_train, cv=k_folds)
print("Neural Network accuracy: {:.2f}%".format(ann_result.mean()*100))

# SVM
svm_model = sk_pipeline.Pipeline([("scaler", sk_preprocessing.StandardScaler()),
                                  ("model", sk_svm.SVC(probability=True))])
svm_result = sk_model_selection.cross_val_score(svm_model, X_train, y_train, cv=k_folds)
print("SVM accuracy: {:.2f}%".format(svm_result.mean()*100))

# Random Forest
rfc_model = sk_ensemble.RandomForestClassifier(n_jobs=-1)
rfc_result = sk_model_selection.cross_val_score(rfc_model, X_train, y_train, cv=k_folds)
print("RFC accuracy: {:.2f}%".format(rfc_result.mean()*100))

# Ensemble
estimators = []
estimators.append(("ANN", ann_model))
estimators.append(("SVM", svm_model))
estimators.append(("RFC", rfc_model))

meta_estimator = sk_linear.LogisticRegression()

ensemble_model = sk_ensemble.StackingClassifier(estimators=estimators, final_estimator=meta_estimator)
ensemble_model.fit(X_train, y_train)
ensemble_prediction = ensemble_model.predict(X_test)

print("\nStacking accuracy: {:.2f}%".format(sk_metrics.accuracy_score(y_test, ensemble_prediction)*100))
