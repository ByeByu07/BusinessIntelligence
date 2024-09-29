import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into DataFrame
def load_data(csv_file):
    return pd.read_csv(csv_file)

# Display the table
def display_table(df):
    print(df)

def count_and_visualize(df, column):
    # Count the occurrences of unique values in the specified column
    counts = df[column].value_counts()
    
    # Display counts in the console
    print(counts)
    
    # Bar chart visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(counts.index, counts.values, color='skyblue')  # Bar chart
    plt.xlabel(f'Unique Values in {column}', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Count of Unique Values in {column}', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels if needed

    for bar in bars:
        yval = bar.get_height()  # Get the height of each bar
        plt.text(bar.get_x() + bar.get_width()/2, yval/2, str(yval), ha='center', va='center', fontsize=10, color='black') 

    plt.tight_layout()
    plt.show()
    
    # Pie chart visualization (optional)
    # plt.figure(figsize=(8, 8))
    # plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors,
    #         textprops={'fontsize': 12})
    # plt.title(f'Pie Chart of {column}', fontsize=16)
    # plt.show()

# Filter data based on multiple conditions and create a pie chart
def create_filtered_pie_chart(df):
    # Filter the data
    filtered_df = df[(df['Ukr'] == 'S') & (df['Wrn'] == 'Merah') & (df['TP'] == 'Kaos')]
    
    # Count the occurrences of 'Nu' (or another column) after filtering
    counts = filtered_df['Nu'].value_counts()  # Replace 'Nu' with the column you want to count
    
    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 12}, colors=plt.cm.Paired.colors)
    
    # Add a title
    plt.title('Filtered Pie Chart (Ukr: S, Wrn: Merah, TP: Kaos)', fontsize=16)
    
    # Show the pie chart
    plt.show()

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = 'garment.csv'  # Replace with the path to your CSV file
    
    # Load data
    df = load_data(csv_file_path)
    
    # Display table
    display_table(df)
    
    # Create pie chart based on filtered data
    count_and_visualize(df,'TP')
    count_and_visualize(df,'Kn')
    count_and_visualize(df,'Nu')
    count_and_visualize(df,'Wrn')
    count_and_visualize(df,'Ukr')
    count_and_visualize(df,'Tgl Msk')
    count_and_visualize(df,'Umur')
    count_and_visualize(df,'Awal')
    count_and_visualize(df,'Masuk')
    count_and_visualize(df,'Jual')
    count_and_visualize(df,'Kirim')
    count_and_visualize(df,'Akhir')
    count_and_visualize(df,'STR')
    count_and_visualize(df,'SSR')
    # create_filtered_pie_chart(df)
