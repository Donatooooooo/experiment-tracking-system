from data.dataset import Dataset
import mlflow, pandas as pd, numpy as np
from sklearn.base import BaseEstimator
from keras.src.models import Model as KerasModel
from mlflow.models.signature import infer_signature

def detect_problem_type(y):
    """
    Detect if problem is a regression or classification from y_test.
    """
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y_values = y.values
    else:
        y_values = y

    if np.issubdtype(y_values.dtype, np.number) and len(np.unique(y_values)) > 10:
        return "regression"
    else:
        return "classification"

def trainAndLog(dataset : Dataset, trainer, experimentName, datasetName,  model_name, tags : dict = None):
    """
    Manages the training of the model within an MLFlow run by 
    logging information, training parameters, and evaluation metrics.
    """
    
    if not mlflow.get_experiment_by_name(experimentName):
        mlflow.create_experiment(experimentName)

    mlflow.set_experiment(experimentName)
    
    with mlflow.start_run():
        # tags log
        if tags is not None:
            for title, tag in tags.items():
                mlflow.set_tag(title, tag)
        
        # dataset logs
        rawdata = mlflow.data.from_pandas(dataset.getDataset(), name = datasetName)
        mlflow.log_input(rawdata, context="training")
        
        # Search for and log hyperparameters
        trainer.findBestParams()
        metrics = trainer.getParams()
        mlflow.log_params(metrics)

        # model training
        trainer.run()

        # model type log
        model = trainer.getModel().__class__.__name__
        mlflow.set_tag("estimator_name", model)

        # metrics log
        mlflow.log_metrics(trainer.getMetrics())

        # Register the trained model and its information
        X_test = trainer.getX()
        model = trainer.getModel()
        
        problem = detect_problem_type(trainer.getY())
        if  problem == 'classification':
            accuracy = metrics.get('accuracy', 0.0)
            if accuracy >= 0.8:
                toRegistry = True
        elif problem == 'regression':
            mse = metrics.get('mse', 1.0)
            if mse <= 0.2:
                toRegistry = True
        
        if toRegistry:
            signature = infer_signature(X_test, model.predict(X_test))
            if isinstance(model, BaseEstimator):
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="Model_Info",
                        signature=signature,
                        input_example=X_test,
                        registered_model_name=model_name
                    )
            elif isinstance(model, KerasModel):
                mlflow.keras.log_model(
                    model=model,
                    artifact_path="Model_Info",
                    signature=signature,
                    registered_model_name=model_name
                )

    mlflow.end_run()
    return None