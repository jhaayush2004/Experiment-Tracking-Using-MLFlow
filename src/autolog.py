# This file is dediated to show that MLFlow can automatically manage all logings without doing them explicitly like eg: mlflow.log_metric('accuracy', accuracy)


import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#this will enable us to generate sharable link where other people can see experiments and vive versa and if 
# import dagshub
# dagshub.init(repo_owner = "Ayush Shaurya Jha", repo_name = "Experiment-Tracking-Using-MLFlow")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

# Mention your experiment below, if we not define name explicitly then it will be stored in default experiment in mlflow
#mlflow.set_experiment('YT-MLOPS-Exp1')

# Mention your experiment below
mlflow.autolog()
mlflow.set_experiment('YT-MLOPS-Exp1')

#in start_run() we can pass experiment_id if we don't want to  mention set_experiment...
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth)
    # mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    # mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__) # only files we have to log it explicitly

    # tags
    mlflow.set_tags({"Author": 'Ayush Shaurya Jha', "Project": "Wine Classification"})

    # Log the model, though not needed to do explicitly
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(accuracy)

# autolog actually logs in many unrequired details also, that why we also went earlier for manual explicitly log mentionings. it logs datasets like train and test data as well as create some tags by own and many more.