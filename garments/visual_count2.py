import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data into DataFrame
def load_data(csv_file):
    return pd.read_csv(csv_file)

# Display the table
def display_table(df):
    print(df)

# Visualize relationships using pair plot, heatmap, and other plots
def visualize_relationships(df):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Pair Plot for numeric columns
    if len(numeric_cols) > 1:  # Ensure there are at least 2 numeric columns
        plt.figure(figsize=(10, 8))
        sns.pairplot(df[numeric_cols])
        plt.suptitle('Pair Plot of Numeric Columns', y=1.02)
        plt.show()
    
    # Correlation Heatmap
    if len(numeric_cols) > 1:  # Ensure there are at least 2 numeric columns
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()  # Compute the correlation matrix for numeric columns
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap')
        plt.show()

    # Box Plots for categorical vs. numeric
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_col, y=num_col, data=df)
            plt.title(f'Box Plot of {num_col} by {cat_col}')
            plt.xticks(rotation=45)
            plt.show()

    # Count Plots for categorical columns
    for cat_col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=cat_col, data=df)
        plt.title(f'Count Plot of {cat_col}')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = 'garment.csv'  # Replace with the actual path to your CSV file
    
    # Load data
    df = load_data(csv_file_path)
    
    # Display table
    display_table(df)
    
    # Visualize relationships
    visualize_relationships(df)
