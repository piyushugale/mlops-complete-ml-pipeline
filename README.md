# mlops-complete-ml-pipeline
This project develops end to end MLOps pipeline utilizing DVC

# Git hub project 
https://github.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline

# Youtube link
https://www.youtube.com/watch?v=SMt3T-K2b_4&list=PLupK5DK91flV45dkPXyGViMLtHadRr6sp&index=5

# MLOps Pipeline

A] data_ingestion.py 

   1. Load parameters from params.yaml file.
   2. load_data(data_url='https://raw.github.com/data/spam.csv')    -- load data from URL
   3. preprocess_data(df)                                           -- drop columns, rename columns, save dataframe
   4. save_data(train_data, test_data, data_path='./data')          -- save train and test data to data_path 

    Output -->
    > python .\src\data_ingestion.py
    2025-02-16 21:23:42,019 - data_ingestion - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 21:23:43,283 - data_ingestion - DEBUG - Data loaded from https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv
    2025-02-16 21:23:43,299 - data_ingestion - DEBUG - Data preprocessing completed
    2025-02-16 21:23:43,318 - data_ingestion - DEBUG - Train and test data saved to ./data\raw
    

B] preprocessing.py

  1. transform_text()                                               -- Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
  2. preprocess_df()                                                -- Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
  3. main()                                                         -- Main function to load raw data, preprocess it, and save the processed data.

    Output -->
    > python .\src\preprocessing.py
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\piyus\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     C:\Users\piyus\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping tokenizers\punkt_tab.zip.
    2025-02-16 21:19:53,322 - data_preprocessing - DEBUG - Data loaded properly
    2025-02-16 21:19:53,322 - data_preprocessing - DEBUG - Starting preprocessing for DataFrame
    2025-02-16 21:19:53,322 - data_preprocessing - DEBUG - Target column encoded
    2025-02-16 21:19:53,322 - data_preprocessing - DEBUG - Duplicates removed
    2025-02-16 21:20:01,795 - data_preprocessing - DEBUG - Text column transformed
    2025-02-16 21:20:01,795 - data_preprocessing - DEBUG - Starting preprocessing for DataFrame
    2025-02-16 21:20:01,795 - data_preprocessing - DEBUG - Target column encoded
    2025-02-16 21:20:01,795 - data_preprocessing - DEBUG - Duplicates removed
    2025-02-16 21:20:04,381 - data_preprocessing - DEBUG - Text column transformed
    2025-02-16 21:20:04,403 - data_preprocessing - DEBUG - Processed data saved to ./data\interim


C] feature_engineering.py

  1. Load parameters from a YAML file.
  2. Load data from a CSV file.
  3. Apply TfIdf to the data. Enhancing features of existing data
  4. Save the dataframe to a CSV file

    Output -->
    > python .\src\feature_engineering.py
    2025-02-16 21:33:07,161 - feature_engineering - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 21:33:07,184 - feature_engineering - DEBUG - Data loaded and NaNs filled from ./data/interim/train_processed.csv
    2025-02-16 21:33:07,184 - feature_engineering - DEBUG - Data loaded and NaNs filled from ./data/interim/test_processed.csv
    2025-02-16 21:33:07,237 - feature_engineering - DEBUG - tfidf applied and data transformed
    2025-02-16 21:33:07,297 - feature_engineering - DEBUG - Data saved to ./data\processed\train_tfidf.csv
    2025-02-16 21:33:07,312 - feature_engineering - DEBUG - Data saved to ./data\processed\test_tfidf.csv    


D] model_building.py

  1. Load parameters from a YAML file.
  2. Load data from a CSV file.
  3. Train the RandomForest model.
  4. Save the trained model to a file.

    Output -->
    > python .\src\model_building.py     
    2025-02-16 21:40:01,759 - model_building - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 21:40:01,775 - model_building - DEBUG - Data loaded from ./data/processed/train_tfidf.csv with shape (4152, 36)
    2025-02-16 21:40:01,775 - model_building - DEBUG - Initializing RandomForest model with parameters: {'n_estimators': 22, 'random_state': 2}
    2025-02-16 21:40:01,775 - model_building - DEBUG - Model training started with 4152 samples
    2025-02-16 21:40:01,864 - model_building - DEBUG - Model training completed
    2025-02-16 21:40:01,868 - model_building - DEBUG - Model saved to models/model.pkl  


E] model_evaluation.py

   1. Load parameters from a YAML file.
   2. Load the trained model from a file.
   3. Load data from a CSV file.
   4. Evaluate the model and return the evaluation metrics.
   5. Save the evaluation metrics to a JSON file.

   Output -->
   > python .\src\model_evaluation.py
   2025-02-16 21:46:25,260 - model_evaluation - DEBUG - Parameters retrieved from params.yaml
   2025-02-16 21:46:25,348 - model_evaluation - DEBUG - Model loaded from ./models/model.pkl
   2025-02-16 21:46:25,366 - model_evaluation - DEBUG - Data loaded from ./data/processed/test_tfidf.csv
   2025-02-16 21:46:25,385 - model_evaluation - DEBUG - Model evaluation metrics calculated
   Initialized DVC repository.

   You can now commit the changes to git.

   WARNING: The following untracked files were present in the workspace before saving but will not be included in the experiment commit:
           params.yaml, projectflow.txt, data/interim/test_processed.csv, data/interim/train_processed.csv, data/processed/test_tfidf.csv, data/processed/train_tfidf.csv, data/raw/test.csv, data/raw/train.csv, experiments/mynotebook.ipynb, experiments/spam.csv, models/model.pkl, src/data_ingestion.py, src/feature_engineering.py, src/model_building.py, src/model_evaluation.py, src/preprocessing.py
   2025-02-16 21:46:28,268 - model_evaluation - DEBUG - Metrics saved to reports/metrics.json


git add .  
git commit -m "MLOps pipeline ready with all components"   
git push origin main  

# Control pipeline stages with DVC.YAML file  
add dvc.yml -- all stagess of pipeline with command, dependency, params, output   
dvc init   

    Output -->
    > dvc init
    ERROR: failed to initiate DVC - '.dvc' exists. Use `-f` to force.
    
    > dvc init -f
    Initialized DVC repository.
    You can now commit the changes to git.
    +---------------------------------------------------------------------+
    |                                                                     |
    |        DVC has enabled anonymous aggregate usage analytics.         |
    |     Read the analytics documentation (and how to opt-out) here:     |
    |             <https://dvc.org/doc/user-guide/analytics>              |
    |                                                                     |
    +---------------------------------------------------------------------+
    What's next?
    ------------
    - Check out the documentation: <https://dvc.org/doc>
    - Get help and share ideas: <https://dvc.org/chat>
    - Star us on GitHub: <https://github.com/iterative/dvc>


dvc repro  -- runs the pipeline with given params  

    Output -->
    > dvc repro
    Stage 'data_ingestion' didn't change, skipping
    Running stage 'data_preprocessing':
    > python src/data_preprocessing.py
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\piyus\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     C:\Users\piyus\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    2025-02-16 22:17:02,913 - data_preprocessing - DEBUG - Data loaded properly
    2025-02-16 22:17:02,913 - data_preprocessing - DEBUG - Starting preprocessing for DataFrame
    2025-02-16 22:17:02,928 - data_preprocessing - DEBUG - Target column encoded
    2025-02-16 22:17:02,933 - data_preprocessing - DEBUG - Duplicates removed
    2025-02-16 22:17:12,082 - data_preprocessing - DEBUG - Text column transformed
    2025-02-16 22:17:12,082 - data_preprocessing - DEBUG - Starting preprocessing for DataFrame
    2025-02-16 22:17:12,084 - data_preprocessing - DEBUG - Target column encoded
    2025-02-16 22:17:12,086 - data_preprocessing - DEBUG - Duplicates removed
    2025-02-16 22:17:18,285 - data_preprocessing - DEBUG - Text column transformed
    2025-02-16 22:17:18,308 - data_preprocessing - DEBUG - Processed data saved to ./data\interim
    Updating lock file 'dvc.lock'

    Running stage 'feature_engineering':
    > python src/feature_engineering.py
    2025-02-16 22:17:25,236 - feature_engineering - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 22:17:25,262 - feature_engineering - DEBUG - Data loaded and NaNs filled from ./data/interim/train_processed.csv
    2025-02-16 22:17:25,275 - feature_engineering - DEBUG - Data loaded and NaNs filled from ./data/interim/test_processed.csv
    2025-02-16 22:17:25,418 - feature_engineering - DEBUG - tfidf applied and data transformed
    2025-02-16 22:17:25,646 - feature_engineering - DEBUG - Data saved to ./data\processed\train_tfidf.csv
    2025-02-16 22:17:25,717 - feature_engineering - DEBUG - Data saved to ./data\processed\test_tfidf.csv
    Updating lock file 'dvc.lock'

    Running stage 'model_building':
    > python src/model_building.py
    2025-02-16 22:17:31,468 - model_building - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 22:17:31,503 - model_building - DEBUG - Data loaded from ./data/processed/train_tfidf.csv with shape (4152, 36)
    2025-02-16 22:17:31,503 - model_building - DEBUG - Initializing RandomForest model with parameters: {'n_estimators': 22, 'random_state': 2}
    2025-02-16 22:17:31,503 - model_building - DEBUG - Model training started with 4152 samples
    2025-02-16 22:17:31,732 - model_building - DEBUG - Model training completed
    2025-02-16 22:17:31,740 - model_building - DEBUG - Model saved to models/model.pkl
    Updating lock file 'dvc.lock'

    Running stage 'model_evaluation':
    > python src/model_evaluation.py
    2025-02-16 22:17:37,747 - model_evaluation - DEBUG - Parameters retrieved from params.yaml
    2025-02-16 22:17:38,040 - model_evaluation - DEBUG - Model loaded from ./models/model.pkl
    2025-02-16 22:17:38,055 - model_evaluation - DEBUG - Data loaded from ./data/processed/test_tfidf.csv
    2025-02-16 22:17:38,101 - model_evaluation - DEBUG - Model evaluation metrics calculated
    WARNING:dvclive:Ignoring `save_dvc_exp` because `dvc repro` is running.
    Use `dvc exp run` to save experiment.
    2025-02-16 22:17:38,844 - model_evaluation - DEBUG - Metrics saved to reports/metrics.json
    Updating lock file 'dvc.lock'

    To track the changes with git, run:

            git add dvc.lock

    To enable auto staging, run:

            dvc config core.autostage true
    Use `dvc push` to send your updates to remote storage.  


# View Dependency Graph with > dvc dag

    > dvc dag
    WARNING: Unable to find `less` in the PATH. Check out <https://man.dvc.org/pipeline/show> for more info.
    +----------------+     
    | data_ingestion |     
    +----------------+     
                *
                *
                *
    +--------------------+   
    | data_preprocessing |   
    +--------------------+   
                *
                *
                *
    +---------------------+
    | feature_engineering |
    +---------------------+
                *
                *
                *
    +----------------+
    | model_building |
    +----------------+
                *
                *
                *
    +------------------+
    | model_evaluation |
    +------------------+


![image](https://github.com/user-attachments/assets/8beacd78-4a0e-4c87-90ba-193b9b02ae4a)  

# Using dvclive
  
pip install dvclive 
  
Change the values in params.yml - 1  > dvc repro  
Change the values in params.yml - 2  > dvc repro  
Change the values in params.yml - 3  > dvc repro  
Change the values in params.yml - 4  > dvc repro  
Change the values in params.yml - 5  > dvc repro  

  > dvc exp run -- creates random named experiment

    Reproducing experiment 'stray-vlei'
    Buildingworkspaceindex |17.0 [00:00, 1.07kentry/s] 
    Comparingindexes       |15.0 [00:00, 1.31kentry/s] 
    Applyingchanges        |0.00 [00:00,     ?file/s] 
    Stage 'data_ingestion' didn't change, skipping
    Stage 'data_preprocessing' didn't change, skipping
    Stage 'feature_engineering' didn't change, skipping
    Stage 'model_building' didn't change, skipping
    Stage 'model_evaluation' didn't change, skipping

    Ran experiment(s): stray-vlei
    Experiment results have been applied to your workspace.

  > dvc exp run

    Reproducing experiment 'weeny-seam'
    Buildingworkspaceindex |17.0 [00:00,  997entry/s]
    Comparingindexes       |15.0 [00:00, 15.0kentry/s]
    Applying changes  ...  



