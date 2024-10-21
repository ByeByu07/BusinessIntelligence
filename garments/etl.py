import pandas as pd

# 1. Extract (Ekstrak)
def extract_data(csv_file):
    return pd.read_csv(csv_file)

# 2. Transform (Transformasi)
def transform_data(df):
     # Ubah kolom 'Tgl Msk' menjadi tipe datetime
    df['Tgl Msk'] = pd.to_datetime(df['Tgl Msk'], format='%m/%d/%Y', errors='coerce')
    
    # Cek apakah ada nilai NaT setelah konversi
    if df['Tgl Msk'].isnull().any():
        print("Warning: Some dates could not be parsed and are set to NaT.")
    
    # Ubah kolom 'Umur' menjadi tipe numerik
    df['Umur'] = pd.to_numeric(df['Umur'].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    
    # Ubah kolom 'Jual', 'Kirim', dan 'Akhir' menjadi tipe numerik
    df['Jual'] = pd.to_numeric(df['Jual'].astype(str).str.strip(), errors='coerce').fillna(0)
    df['Kirim'] = pd.to_numeric(df['Kirim'].astype(str).str.strip(), errors='coerce').fillna(0)
    df['Akhir'] = pd.to_numeric(df['Akhir'].astype(str).str.strip(), errors='coerce').fillna(0)

    # Hitung total dari kolom 'Jual', 'Kirim', 'Akhir'
    df['Total'] = df[['Jual', 'Kirim', 'Akhir']].sum(axis=1)
    
    # Menghapus kolom yang tidak perlu (jika ada)
    df.drop(['STR', 'SSR'], axis=1, inplace=True, errors='ignore')

    return df

# 3. Load (Muat)
def load_data(df, output_file):
    df.to_csv(output_file, index=False)
    print("ETL process completed. Transformed data saved to:", output_file)

