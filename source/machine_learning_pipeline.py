import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from definitions import NORMALIZED_DIR_PATH
from PIL import Image


class MachineLearningPipeline:
    @staticmethod
    def load_normalized_images():
        images = []
        labels = []

        label_mapping = {'EE': 0, 'K': 1}

        for file in os.listdir(NORMALIZED_DIR_PATH):
            if file.endswith('_heatmap.png'):
                file_path = os.path.join(NORMALIZED_DIR_PATH, file)
                label = None

                # Determine the label based on the file name
                if file.startswith('EE'):
                    label = label_mapping['EE']
                elif file.startswith('K'):
                    label = label_mapping['K']

                if label is not None:  # Proceed if a valid label was found
                    try:
                        image = Image.open(file_path).convert('L')  # Convert to grayscale
                        image_np = np.array(image).flatten() / 255.0  # Normalize pixel values to [0, 1]
                        images.append(image_np)
                        labels.append(label)
                    except IOError:
                        print(f"Could not read image: {file_path}")
                else:
                    print(f"File {file} does not match labeling convention.")

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int8)


    @staticmethod
    def run_pipeline():
        processed_images, processed_labels = MachineLearningPipeline.load_normalized_images()

        results_df = pd.DataFrame(columns=['Replication', 'Model', 'Best_Parameters', 'Accuracy', 'F1_Score',
                                           'Classification_Report'])
        iteration_results = pd.DataFrame(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                                  'param_classifier__max_depth', 'param_classifier__n_estimators',
                                                  'param_smote__k_neighbors', 'params', 'split0_test_score',
                                                  'split1_test_score', 'split2_test_score', 'split3_test_score',
                                                  'split4_test_score', 'mean_test_score', 'std_test_score',
                                                  'rank_test_score', 'Model', 'Replication'])
        results_df.to_csv('ml_results.csv', header=True, index=False)
        iteration_results.to_csv('detailed_ml_results.csv', header=True, index=False)
        # detailed_results_df = pd.DataFrame()

        models = [
            ('RandomForest', RandomForestClassifier(random_state=42), {
                'smote__k_neighbors': [1, 3, 5],
                'classifier__n_estimators': [100, 150, 200],
                'classifier__max_depth': [5, 10, 15]
            }),
            ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000), {
                'smote__k_neighbors': [1, 3, 5]
                # 'classifier__C': [0.01, 0.1, 1]
                # 'classifier__solver': ['liblinear', 'sag', 'lbfgs']
            }),
            ('SVM', SVC(random_state=42), {
                'smote__k_neighbors': [1, 3, 5],
                'classifier__C': [0.01, 0.1, 1],
                # 'classifier__kernel': ['linear', 'rbf']
            })
        ]

        num_replications = 2
        for replication in range(num_replications):

            print("Replication " + str(replication))
            X_train, X_test, y_train, y_test = train_test_split(processed_images, processed_labels, test_size=0.3, stratify=processed_labels)

            for model_name, model, parameters in models:
                pipeline = Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', model)
                ])

                clf = GridSearchCV(pipeline, parameters, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
                clf.fit(X_train, y_train)

                iteration_results = pd.DataFrame(clf.cv_results_)
                iteration_results['Model'] = model_name
                iteration_results['Replication'] = replication + 1
                # detailed_results_df = pd.concat([detailed_results_df, iteration_results], ignore_index=True)
                iteration_results.to_csv('detailed_ml_results.csv', mode='a', header=False, index=False)
                # detailed_results_df.to_excel('detailed_ml_results.xlsx', index=False)

                # Collect summary results
                best_params = clf.best_params_
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary')
                classification_rep = json.dumps(classification_report(y_test, y_pred, output_dict=True))

                new_row = {
                    'Replication': replication + 1,
                    'Model': model_name,
                    'Best_Parameters': str(best_params),
                    'Accuracy': accuracy,
                    'F1_Score': f1,
                    'Classification_Report': classification_rep
                }
                new_row_df = pd.DataFrame([new_row])

                new_row_df.to_csv('ml_results.csv', mode='a', header=False, index=False)
                # results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                # results_df.to_excel('ml_results.xlsx', index=False)


        # print(results_df.mean(numeric_only=True))
        # results_df.to_excel('ml_results.xlsx', index=False)
        # detailed_results_df.to_excel('detailed_ml_results.xlsx', index=False)
