import pandas as pd
from supabase_config import supabase

def load(df: pd.DataFrame, table_name: str, save_option: str):
    if save_option == 'api':
        # Iterate over the DataFrame and insert each row into Supabase
        for index, row in df.iterrows():
            # Convert row to dictionary
            data_dict = row.to_dict()

            # Insert the data into Supabase table
            response = supabase.table(table_name).insert(data_dict).execute()

            if response.status_code == 201:
                print(f"Inserted row {index} successfully")
            else:
                print(f"Failed to insert row {index}. Error: {response.json()}")
    
    elif save_option == 'csv':
        # Save DataFrame to CSV
        csv_file_path = f"{table_name}.csv"  # You can modify the filename if needed
        df.to_csv(csv_file_path, index=False)
        print(f"DataFrame saved to {csv_file_path}")
    
    else:
        print("Invalid save option. Please use 'api' or 'csv'.")

# Example usage
# load(df, "steam", "api")  # Save to Supabase API
# load(df, "steam", "csv")  # Save to CSV
