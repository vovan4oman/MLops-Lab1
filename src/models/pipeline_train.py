import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os

def train_pipeline():
    mlflow.set_experiment("Pipeline_Experiment")

   
    df = pd.read_csv('data/raw/dataset.csv')

    
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('class', axis=1)
    y = df['class']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])

        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

    
        pipeline.fit(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)

        
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(pipeline, "pipeline_model")

        print(f"Pipeline успішно навчено!")
        print(f"Крос-валідація (середня точність): {cv_mean:.4f}")
        print(f"Тестова точність: {test_acc:.4f}")

       
        os.makedirs('models', exist_ok=True)
        with open('models/pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

if __name__ == "__main__":
    train_pipeline()