import read_dataset as rd

if "__name__" == "__main__":
    df = rd.read('data_full_1.3.parquet', './Materiale/dataset')
    print(df)