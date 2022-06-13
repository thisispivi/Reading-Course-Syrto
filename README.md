# Reading-Course-Syrto
Code of the regressors for the syrto project.

To see how to generate the reports check the [link](REPORT_GENERATOR.md).

# Index

- [Folder structure](#folder-structure)
- [How to run the project](#how-to-run-the-project)
  * [Parameters configuration](#parameters-configuration)
  * [Utils Configuration](#utils-configuration)
  * [Run the code](#run-the-code)

# Folder structure
```
.gitignore
eda.py
metrics.py   ### File with all the functions to compute the metrics
README.md
read_dataset.py   ### File with the function to read the dataset
regression.py   ### File that performs the regression
regressors.py   ### FIle with all the regressors
REPORT_GENERATOR.MD
utils.py   ### File with useful fumctions
__init__.py   ### Main
         
+---dataset   ### Here insert the datasets
            
+---Documentazione    ### Doc
        Elenco materiale.pdf
        Presentazione Syrto.pdf
        Step2.pdf
        Time Series Analysis Forecasting.pdf
        
+---export   ### Folder in which the exports will be saved

+---latex   ### Folder with the program to generate latex reports
    
        edit.py
        
    +---benchmark
    +---regressors
    +---tft
            
+---syrto   ### Syrto functions
        pytorch_forecasting.py
        utils.py
        __init__.py
```

# How to run the project
The main file is ```__init__.py```. The file has many parameters to configure before the run. Basically the code will take the dataset
and for each field it will create a new column with the value of the field for the next year (1 year predictions) or for the next
two years (2 year prediction).

So in the dataset for each year there will be another column with the value

## Parameters configuration
1. Logspace: *(bool)* &#8594; **True** use logspace / **False** otherwise
```python
logspace = True
```
2. Csv: *(bool)* &#8594; **True** use the previous generated csv file / **False** import the dataset from ```data_4.0.csv``` and generate new csv files.  
**IMPORTANT: if you created a file csv using 1 year predictions, if you want to run 2 years predictions you have to set csv to False to generate new csv file**
```python
csv = True
```
3. Benchmark: *(bool)* &#8594; **True** perform benchmark analysis / **False** Perform the regression  
```python
benchmark = True
```

4. Targets: *(List of str)* Columns of the file
```python
targets = ["TOTALE IMMOB IMMATERIALI",
           "TOTALE IMMOB MATERIALI",
           "TOTALE IMMOB FINANZIARIE",
           "TOTALE RIMANENZE",
           "ATTIVO CIRCOLANTE",
           "TOTALE CREDITI",
           "Capitale sociale",
           "TOTALE PATRIMONIO NETTO",
           "DEBITI A BREVE",
           "DEBITI A OLTRE",
           "TOTALE PASSIVO",
           "TOT VAL DELLA PRODUZIONE",
           "Ricavi vendite e prestazioni",
           "COSTI DELLA PRODUZIONE",
           "RISULTATO OPERATIVO",
           "TOTALE PROVENTI E ONERI FINANZIARI",
           "TOTALE PROVENTI/ONERI STRAORDINARI",
           "RISULTATO PRIMA DELLE IMPOSTE",
           "Totale Imposte sul reddito correnti, differite e anticipate",
           "UTILE/PERDITA DI ESERCIZIO",
           "EBITDA",
           "Capitale circolante netto",
           "Materie prime e consumo",
           "Totale costi del personale",
           "TOT Ammortamenti e svalut",
           "Valore Aggiunto"]
```

5. Key: *(List of str)* The keys
```python
key = ["id",
       "bilancio_year"]
```

6. All Fields: *(bool)* &#8594; **True** To perform regression for all fields / **False** In another variable you will declare the field to predict  
```python
all_fields = True
```

7. Select field: *(str)* &#8594; In case ```all_fields``` is false, select a field to predict. Uncomment to select one
```python
# field_name = "future_TOTALE_IMMOB_IMMATERIALI"
# field_name = "future_TOTALE_IMMOB_MATERIALI"
# field_name = "future_TOTALE_IMMOB_FINANZIARIE"
# field_name = "future_TOTALE_RIMANENZE"
# field_name = "future_ATTIVO_CIRCOLANTE"
# field_name = "future_TOTALE_CREDITI"
# field_name = "future_Capitale_sociale"
# field_name = "future_TOTALE_PATRIMONIO_NETTO"
# field_name = "future_DEBITI_A_BREVE"
# field_name = "future_DEBITI_A_OLTRE"
# field_name = "future_TOTALE_PASSIVO"
# field_name = "future_TOT_VAL_DELLA_PRODUZIONE"
# field_name = "future_Ricavi_vendite_e_prestazioni"
# field_name = "future_COSTI_DELLA_PRODUZIONE"
# field_name = "future_RISULTATO_OPERATIVO"
# field_name = "future_TOTALE_PROVENTI_E_ONERI_FINANZIARI"
# field_name = "future_TOTALE_PROVENTI_ONERI_STRAORDINARI"
# field_name = "future_RISULTATO_PRIMA_DELLE_IMPOSTE"
# field_name = "future_Totale_Imposte_sul_reddito_correnti_differite_e_anticipate"
# field_name = "future_UTILE_PERDITA_DI_ESERCIZIO"
# field_name = "future_EBITDA"
# field_name = "future_Capitale_circolante_netto"
# field_name = "future_Materie_prime_e_consumo"
# field_name = "future_Totale_costi_del_personale"
# field_name = "future_TOT_Ammortamenti_e_svalut"
field_name = "future_Valore_Aggiunto"
```

8. Select the regressors &#8594; **True** to select / **False** the opposite
```python
regressors_list = {
    "ols": False,  # Ordinary Least Squares
    "ridge": False,  # Ridge Regressor
    "lasso": False,  # Lasso Regressor
    "elastic": False,  # Elastic Net Regressor
    "lars": True,  # Lars Regressor
    "bayesian": True,  # Bayesian Regressor
    "stochastic": False,  # Stochastic Gradient Descent Regressor
    "passive": False,  # Passive Aggressive Regressor
    "kernel": False,  # Kernel Ridge Regressor
    "svr": False,  # Support Vector Regressor
    "nn": False,  # Nearest Neighbour Regressor
    "gauss": False,  # Gaussian Process Regressor
    "decision": False,  # Decision Tree Regressor
    "random": False,  # Random Forest Regressor
    "ada": True,  # Ada Boost Regressor
    "gradient": False,  # Gradient Boost Regressor
    "ensemble": False  # Ensemble Regressor
}
```

9. Field Array: *(List of str)* All the possible fields
```python
field_array = ["future_TOTALE_IMMOB_IMMATERIALI",
               "future_TOTALE_IMMOB_MATERIALI",
               "future_TOTALE_IMMOB_FINANZIARIE",
               "future_TOTALE_RIMANENZE",
               "future_ATTIVO_CIRCOLANTE",
               "future_TOTALE_CREDITI",
               "future_Capitale_sociale",
               "future_TOTALE_PATRIMONIO_NETTO",
               "future_DEBITI_A_BREVE",
               "future_DEBITI_A_OLTRE",
               "future_TOTALE_PASSIVO",
               "future_TOT_VAL_DELLA_PRODUZIONE",
               "future_Ricavi_vendite_e_prestazioni",
               "future_COSTI_DELLA_PRODUZIONE",
               "future_RISULTATO_OPERATIVO",
               "future_TOTALE_PROVENTI_E_ONERI_FINANZIARI",
               "future_TOTALE_PROVENTI_ONERI_STRAORDINARI",
               "future_RISULTATO_PRIMA_DELLE_IMPOSTE",
               "future_Totale_Imposte_sul_reddito_correnti_differite_e_anticipate",
               "future_UTILE_PERDITA_DI_ESERCIZIO",
               "future_EBITDA",
               "future_Capitale_circolante_netto",
               "future_Materie_prime_e_consumo",
               "future_Totale_costi_del_personale",
               "future_TOT_Ammortamenti_e_svalut",
               "future_Valore_Aggiunto"]
```

10. File Name: *(str)* The name of the file. It's possible to use the file name generator or select a particular name
```python
file_name = generate_file_name(field_name, benchmark)
# file_name = "export.csv"  # Uncomment to use custom file names
```

## Utils Configuration

There are some parameters in the file ```utils.py``` that needs to be configured.

1. Add Future Values &#8594; Function that for each field create a new column in the dataframe that will contain the value of that field
for the next year or for the next two years. To choose between the two types of predictions you'll have to uncomment a specific row
```python
def add_future_values(df, targets):
    """
    Adds a column in the dataframe containing the value of each column for the next year

    Args:
        targets:
        df: (Pandas Dataframe) the dataset

    Returns:
        (Pandas Dataframe) The dataset with the new columns
    """

    for t in targets:
        col_name = "future_" + t
        
        ### UNCOMMENT THIS ROW TO SELECT TWO YEARS PREDICTION ###
        df.loc[:, col_name] = df.groupby('id')[t].transform(lambda group: group.shift(-2))
        
        ### UNCOMMENT THIS ROW TO SELECT ONE YEAR PREDICTION ###
        # df.loc[:, col_name] = df.groupby('id')[t].transform(lambda group: group.shift(-1))

    return df
```

2. Split Dataset &#8594; This function will split the dataset in training, validation e test set (this one is not used). 
It's important to notice that the code will consider the future values, so, for example, if we want to perform one year predictions
when we take training set from 2017, we are actually taking the data from 2011  to 2016 with 2017 labels. The same for the validation set.

The code below is the training from 2011 to 2017 with 1 year predictions.
```python
def split_dataset(df):
    """
    Split the dataset in Training Set, Validation Set and Test Set

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        training: (Pandas Dataframe) Training Set
        validation: (Pandas Dataframe) Validation Set
        test: (Pandas Dataframe) Test Set
    """
    training = df[df.bilancio_year < 2017]
    validation = df[df.bilancio_year == 2017]
    test = df[df.bilancio_year == 2020]
    return training, validation, test
```

3. Split Dataset Benhmark &#8594; This function will split the dataset in training and validation set when benchmark mode is active.
It's important to notice that the code will consider the future values, so, for example, if we want to perform two years predictions
when we take training set from 2016, we are actually taking the 2016 data with the 2018 labels. The same for the validation set.

For example this code is for the training of 2018 with 2 years predictions (2020).
```python
def split_dataset_benchmark(df):
    """
    Split the dataset in Training Set year(2016), Validation Set year(2017). This is used for the benchmark

    Args:
        df: (Pandas Dataframe) the dataset

    Returns:
        training: (Pandas Dataframe) Training Set (2016 data)
        validation: (Pandas Dataframe) Validation Set (2017 data)
    """
    training = df[df.bilancio_year == 2016]
    validation = df[df.bilancio_year == 2018]
    return training, validation
```

## Run the code
1. Open a terminal and run this line
```
python __init__.py
```
