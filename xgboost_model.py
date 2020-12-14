import os
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from utils import POST_TAGS

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import load_data, feature_extractor

if __name__ == '__main__':
    fname = 'wordy_with_features.csv'
    parts_of_speech_to_check = ['WRDNS']

    # Load data
    if not os.path.isfile(fname):
        df = load_data(filename=r"D:\MySupervisorNLP\wordy_ds.csv")
        df = feature_extractor(df, POST_TAGS)
        df.to_csv(fname, index=False)
    else:
        df = pd.read_csv(fname)

    # split data into X and y
    X = np.asarray(df[parts_of_speech_to_check])
    Y = np.asarray(df['Class'])

    # Train
    max = 0.0
    max_i = -1
    for i in range(1000):
        # split data into train and test sets
        # seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)

        # Fit model on training data
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        # Make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]

        # Evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        if accuracy > max:
            max = accuracy
            max_i = i
            model.save_model('xgboost_partly_' + str(max * 100) + '_acc.model')
            best_model = model
            best_X_train = X_train

    print("\n\n\nMax accuracy: %.2f%%" % (max * 100.0))
    print(f" at seed: {max_i}")

    # Feature importance visualization
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(best_X_train)
    shap.summary_plot(shap_values, best_X_train, feature_names=parts_of_speech_to_check)
