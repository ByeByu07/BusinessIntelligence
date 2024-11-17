import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import json
import re
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score

def load_and_prepare_data(json_file):
    """Load and prepare data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter out rows with 'N/A' or invalid prices
    df = df[df['price'] != 'N/A']
    df.loc[df['price'] == 'Free To Play', 'price'] = '0'
    
    return df

def extract_price(price_string):
    try:
        price = price_string.replace("Rp", "").replace(" ", "")
        return int(price)
    except:
        return 0.0

def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Convert price to numeric
    df['price_numeric'] = df['price'].apply(extract_price)

    try:
        df['price_category'] = pd.qcut(
            df['price_numeric'], 
            q=4, 
            labels=['budget', 'standard', 'premium', 'deluxe']
        )
    except ValueError:
        bins = [-float('inf')]
        bins.extend(df['price_numeric'].quantile([0.25, 0.5, 0.75, 1.0]).tolist())
        df['price_category'] = pd.cut(
            df['price_numeric'],
            bins=bins,
            labels=['budget', 'standard', 'premium', 'deluxe'],
            include_lowest=True,
            duplicates='drop'
        )
    
    # Create one-hot encoding for tags
    tags_dummies = df['tags'].apply(lambda x: pd.Series([1] * len(x), index=x))
    tags_dummies = tags_dummies.fillna(0)
    
    return tags_dummies, df['price_category']

def perform_feature_selection(X, y, method='mutual_info', k=None):
    """Perform feature selection using either mutual information or chi-squared test"""
    from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
    
    if k is None:
        k = len(X.columns)

    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'chi2':
        selector = SelectKBest(score_func=chi2, k=k)
    else:
        raise ValueError("Method must be either 'mutual_info' or 'chi2'")
    
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    return selector, selected_features, feature_scores

def optimize_k_parameter(X, y, k_range=None):
    """
    Find optimal k parameter for feature selection using cross-validation
    
    Parameters:
    -----------
    X : pandas DataFrame
        The feature matrix
    y : array-like
        Target variable
    k_range : list, optional
        List of k values to try. If None, uses range from 10 to n_features by 10
    """
    if k_range is None:
        max_features = min(len(X.columns), 200)  # Cap at 200 features
        k_range = list(range(10, max_features + 1, 10))
    
    # Initialize variables to store results
    cv_scores = []
    k_values = []
    
    # Base classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    # Scorer
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    # Try different k values
    for k in k_range:
        # Select features
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Perform cross-validation
        scores = cross_val_score(
            rf,
            X_selected,
            y,
            cv=5,
            scoring=f1_scorer,
            n_jobs=-1
        )
        
        # Store results
        cv_scores.append(np.mean(scores))
        k_values.append(k)
        
        print(f"k={k}: CV Score = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Find optimal k
    optimal_k = k_values[np.argmax(cv_scores)]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, marker='o')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Cross-validation Score (F1)')
    plt.title('Feature Selection: k Parameter Optimization')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.grid(True)
    plt.savefig('k_optimization.png')
    plt.close()
    
    return optimal_k, cv_scores, k_values

def train_model(X, y):
    """Train the model using GridSearchCV"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    return grid_search, scaler, X_test_scaled, y_test

def predict_price_category(model, scaler, tags, selected_features):
    """Predict price category for new tags"""
    features = pd.Series(0, index=selected_features)
    for tag in tags:
        if tag in features.index:
            features[tag] = 1
    
    features_scaled = scaler.transform(features.values.reshape(1, -1))
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return prediction[0], probabilities[0]

def main():
    # Load and prepare data
    df = load_and_prepare_data('datas/data_08_34_16_11_2024.json')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Convert target to numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Optimize k parameter
    print("\nOptimizing k parameter...")
    k_range = list(range(10, min(len(X.columns), 384) + 1, 10))
    optimal_k, cv_scores, k_values = optimize_k_parameter(X, y_encoded, k_range)
    print(f"\nOptimal number of features (k): {optimal_k}")
    
    # Feature selection with optimal k
    selector, selected_features, feature_scores = perform_feature_selection(
        X, y_encoded, method='chi2', k=optimal_k
    )
    X_selected = selector.transform(X)
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    # Train model
    grid_search, scaler, X_test_scaled, y_test = train_model(X_selected, y_encoded)
    
    # Calculate price ranges for each category
    price_ranges = df.groupby('price_category')['price_numeric'].agg(['min', 'max'])
    
    # Print results with price ranges
    print("\nPrice Category Ranges:")
    for category in ['budget', 'standard', 'premium', 'deluxe']:
        min_price = f"Rp {price_ranges.loc[category, 'min']:,.0f}"
        max_price = f"Rp {price_ranges.loc[category, 'max']:,.0f}"
        print(f"{category.title()}: {min_price} - {max_price}")
    
    print("\nBest parameters:", grid_search.best_params_)
    print("\nBest cross-validation score:", grid_search.best_score_)
    
    # Make predictions
    y_pred = grid_search.predict(X_test_scaled)
    
    # Convert predictions back to labels
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Print top features
    print(f"\nTop {min(20, optimal_k)} Most Important Tags for Price Prediction:")
    print(feature_scores.head(min(20, optimal_k)))
    
    # Save feature importance
    feature_scores.to_csv('feature_importance.csv', index=False)
    
    # Example prediction
    example_tags = ["Free To Play", "Indie"]
    predicted_category, probabilities = predict_price_category(
        grid_search, scaler, example_tags, selected_features
    )
    
    # Update prediction output to include price range
    predicted_category = le.inverse_transform([predicted_category])[0]
    min_price = f"Rp {price_ranges.loc[predicted_category, 'min']:,.0f}"
    max_price = f"Rp {price_ranges.loc[predicted_category, 'max']:,.0f}"
    
    print(f"\nPredicted price category for {example_tags}: {predicted_category}")
    print(f"Price Range: {min_price} - {max_price}")
    print("Probability distribution:")
    for category, prob in zip(le.classes_, probabilities):
        min_price = f"Rp {price_ranges.loc[category, 'min']:,.0f}"
        max_price = f"Rp {price_ranges.loc[category, 'max']:,.0f}"
        print(f"{category} ({min_price} - {max_price}): {prob:.2f}")

if __name__ == "__main__":
    main()