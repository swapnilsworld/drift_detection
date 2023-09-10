import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# for supervised:
from sklearn.cluster import KMeans
import csv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Read the dataset
data = pd.read_csv("system_monitoring_50k.csv")

# Calculate drift label based on threshold values and differences
threshold_ram = 60  # Example threshold for RAM usage (%)
threshold_cpu = 45  # Example threshold for CPU utilization (%)
difference_threshold_ram = 4  # Example difference threshold for RAM usage (%)
difference_threshold_cpu = 2  # Example difference threshold for CPU utilization (%)

data["drift_label"] = np.where(
    ((data["ram_usage"] > threshold_ram) & (data["cpu_utilization"] > threshold_cpu))
    | (
        (data["ram_usage"].diff() > difference_threshold_ram)
        & (data["cpu_utilization"].diff() > difference_threshold_cpu)
    ),
    1,
    0,
)

# Split the data into features and target
X = data[["ram_usage", "cpu_utilization"]]
y = data["drift_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize classifiers
classifiers = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ("AdaBoost", AdaBoostClassifier(random_state=42)),
    ("Bagging", BaggingClassifier(random_state=42)),
    ("ExtraTrees", ExtraTreesClassifier(random_state=42)),
    # (
    #     "Voting",
    #     VotingClassifier(
    #         estimators=[
    #             ("rf", RandomForestClassifier(random_state=42)),
    #             ("gb", GradientBoostingClassifier(random_state=42)),
    #         ],
    #         voting="soft",
    #     ),
    # ),
    ("SVC", SVC(random_state=42)),
    ("MLP", MLPClassifier(random_state=42)),
    ("KNeighbors", KNeighborsClassifier()),
    ("GaussianNB", GaussianNB()),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
    ("LogisticRegression", LogisticRegression(random_state=42)),
    # # Unsupervised
    # (
    #     "Unsupervised: Isolation Forest",
    #     IsolationForest(contamination=0.1, random_state=42),
    # ),
    # ("Unsupervised: K-Means", KMeans(n_clusters=2, random_state=42)),
    # ("Unsupervised: One-Class SVM", OneClassSVM(nu=0.1)),
]

# Calculate the number of screens needed
num_screens = int(np.ceil(len(classifiers) / 4))

print("Checking for Supervised algorithms")
# Define a list to store the results
results = []
# Loop through screens
for screen in range(num_screens):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Loop through classifiers within each screen
    for row in range(2):
        for col in range(2):
            clf_idx = screen * 4 + row * 2 + col
            if clf_idx < len(classifiers):
                clf_name, clf = classifiers[clf_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"{clf_name} Accuracy: {accuracy:.2f}")
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)

                # Append the results to the list
                results.append([clf_name, accuracy, precision, recall, f1, roc_auc])

                # Generate confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                sns.heatmap(
                    conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[row, col]
                )
                axs[row, col].set_xlabel("Predicted")
                axs[row, col].set_ylabel("Actual")
                axs[row, col].set_title(f"Confusion Matrix for {clf_name}")

    plt.tight_layout()
    plt.show()
    # Define the CSV file name
csv_filename = "classification_results.csv"

# Write the results to a CSV file
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Classifier", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )
    writer.writerows(results)

print(f"Results saved to {csv_filename}")
