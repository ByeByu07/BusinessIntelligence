from transform import transform
from load import load
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Define the path to your JSON file
    path_to_json = "datas\data_08_34_16_11_2024.json"  # Update this path
    unique_field = "title"  # The field you consider as unique

    # Transform the data
    df = transform(path_to_json, unique_field)

    # # Assume 'target' is the column we want to predict
    # X = df.drop(columns=['price'])  # Features
    # y = df['price']  # Target variable
    # X = pd.get_dummies(X, columns=['title','developer', 'publisher','release_date'], drop_first=True)
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Initialize the Random Forest Classifier
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # # Fit the model to the training data
    # rf.fit(X_train, y_train)

    # # Make predictions on the test set
    # y_pred = rf.predict(X_test)

    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    if not df.empty:
        # Load the transformed data into Supabase

        # sample_data = df.head(10)  # Take a sample of 10 rows
        load(df, "steam", 'csv')

        # load(df, "steam","api")  # Replace 'your_table_name' with the target table name in Supabase
    else:
        print("No valid data to insert")

# Execute the script
if __name__ == "__main__":
    main()