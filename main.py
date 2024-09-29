from etl import extract_data,load_data,transform_data
from mongodb import load_csv_to_mongodb

if __name__ == "__main__":

  # configuration
  input_file = 'garment.csv'
  output_file = 'transformed_data.csv'

  print("Running...")

  print("Extract...")
  data = extract_data(input_file)

  print("Transforming...")
  transformed_data = transform_data(data)
  load_data(transformed_data, output_file)

  print("Saving to mongodb...")
  load_csv_to_mongodb(output_file,'garment','orders')
  print("----Done----")
