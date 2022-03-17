# Reading-Course-Syrto

# Index
- [How to run the project](#how-to-run-the-project)
  * [Use the csv file](#use-the-csv-file)
  * [Use the csv file](#use-the-csv-file-1)
  * [Configure the variables](#configure-the-variables)

# How to run the project
There are two ways to start:
1. Use the *parquet* file
2. Use the *csv* file  

After that the method is the same for both methods

---

## Use the csv file
1. Extract the dataset from the ```dataset.rar``` file in the *dataset* folder
2. Change the value of the variable *csv* to True from ```__init__.py``` file (line 8)
```python
csv = True
```

## Use the csv file
1. Change the value of the variable *csv* to False from ```__init__.py``` file (line 8)
```python
csv = False
```

---

## Configure the variables
1. The variable ```all_fields``` is useful if you want to generate the metrics for each column of the csv file. When it 
is *False* you have to select a column to select.
```python
all_fields = True
```
2. The variable ```benchmark``` activate the benchmark mode. This mode will take as prediction the data from 2016 and as
correct values the data from 2017, then it will compute the metrics. This mode won't perform the regression. To perform 
the regression this variable must be *False*.
```python
benchmark = False
```
3. The variable ```targets``` represents the data of the companies that will be taken from the parquet file
```python
targets = ["TOTALE IMMOBILIZZAZIONI",
           "ATTIVO CIRCOLANTE",
           "TOTALE ATTIVO",
           "TOTALE PATRIMONIO NETTO",
           "DEBITI A BREVE",
           "DEBITI A OLTRE",
           "TOTALE DEBITI",
           "TOTALE PASSIVO",
           "TOT VAL PRODUZIONE",
           "RISULTATO OPERATIVO",
           "RISULTATO PRIMA DELLE IMPOSTE",
           "UTILE/PERDITA DI ESERCIZIO"]
```
4. The variable ```keys``` represents the id of the companies that will be taken from the parquet file
```python
key = ["id",
       "bilancio_year"]
```
5. The variable ```regressors_list``` represents all the regressors available. To select one just set it value to True. 
False otherwise. These are the regressors available:
   1. Ordinary Least Squares
   2. Ridge Regressor
   3. Lasso Regressor
   4. Elastic Net Regressor
   5. Lars Regressor
   6. Bayesian Regressor
   7. Stochastic Gradient Descent Regressor
   8. Passive Aggressive Regressor
   9. Kernel Ridge Regressor
   10. Support Vector Regressor
   11. Nearest Neighbour Regressor
   12. Gaussian Process Regressor
   13. Decision Tree Regressor
   14. Random Forest Regressor
   15. Ada Boost Regressor
   16. Gradient Boost Regressor
   17. Ensemble Regressor
```python
# Select the regressors (True to select, False the opposite)
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
    "nn": True,  # Nearest Neighbour Regressor
    "gauss": False,  # Gaussian Process Regressor
    "decision": True,  # Decision Tree Regressor
    "random": True,  # Random Forest Regressor
    "ada": True,  # Ada Boost Regressor
    "gradient": True,  # Gradient Boost Regressor
    "ensemble": False  # Ensemble Regressor
}
```
6. If you chose *all_fields = False* then select the variable ```field_name```. It represents the name of the variable that will be predicted. To select one just 
uncomment one row
```python
field_name = "future_TOTALE_IMMOBILIZZAZIONI"
# field_name = "future_ATTIVO_CIRCOLANTE"
# field_name = "future_TOTALE_ATTIVO"
# field_name = "future_TOTALE_PATRIMONIO_NETTO"
# field_name = "future_DEBITI_A_BREVE"
# field_name = "future_DEBITI_A_OLTRE"
# field_name = "future_TOTALE_DEBITI"
# field_name = "future_TOTALE_PASSIVO"
# field_name = "future_TOT_VAL_PRODUZIONE"
# field_name = "future_RISULTATO_OPERATIVO"
# field_name = "future_RISULTATO_PRIMA_DELLE_IMPOSTE"
# field_name = "future_UTILE_PERDITA_DI_ESERCIZIO"
```
6. The variable ```file_name``` represents the name of the file in which the results of the regression and 
classification will be saved. There is a function that creates the file name. The names will be like:
```
field_name day time.csv

Example: TOTALE_IMMOBILIZZAZIONI 2022-02-14 12-54-48.csv
```

## Run the code
1. Open a terminal and run this line
```
python __init__.py
```
