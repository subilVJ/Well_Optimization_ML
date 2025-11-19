# Well_Optimization_ML
 
 ## Work flow of the project

 1. update Config yaml
 2. update schema yaml
 3. update param yaml
 4. update entity
 5. update the configuration manager in src config
 6. update the components
 7. update the pipeline 
 8. update the main.py
 9. update the app.py

well optimization/README.md


https://dagshub.com/subilVJ/Well_Optimization_ML.mlflow


import dagshub
dagshub.init(repo_owner='subilVJ', repo_name='Well_Optimization_ML', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


  9445f026b81bde037b36dd2c0eaf1eccc269e6ac
