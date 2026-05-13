import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train():
    
    mlflow.set_experiment("Mushroom_Classification")
    
   
    df = pd.read_csv('data/raw/dataset.csv')
    
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    
    
    X = df.drop('class', axis=1)
    y = df['class']
    
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    experiments = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10}
    ]

    for params in experiments:
        with mlflow.start_run(): 
          
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
           
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            print(f"Тренування завершено! Параметри: {params} -> Точність: {acc:.4f}")

if __name__ == "__main__":
    train()