from Dataset.dataset import Dataset
from mlflow.models import infer_signature
import mlflow, mlflow.experiments, dagshub

def trainAndLog(dataset : Dataset, trainer, experimentName, signature,
                    datasetName, tags : dict = None, toRegistry = True):
    """
    Manages the training of the model within an MLFlow run by 
    logging information, training parameters, and evaluation metrics.
    """

    dagshub.init(repo_owner='donatooooooo', repo_name='MLflow_Server', mlflow=True)
    
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
        # mlflow.log_input(rawdata, context="training")
        
        # Search for and log hyperparameters
        trainer.findBestParams()
        # mlflow.log_params(trainer.getParams())

        # model training
        trainer.run()

        # model type log
        model = trainer.getModel().__class__.__name__
        mlflow.set_tag("estimator_name", model)

        # metrics log
        mlflow.log_metrics(trainer.getMetrics())

        if toRegistry:
            # Register the trained model and its information
            X_test = trainer.getX()
            model = trainer.getModel()
            
            if signature == 0:
                mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path = "Model_Info",
                    signature = infer_signature(X_test, model.predict(X_test)),
                    input_example = X_test,
                    registered_model_name = "Istological Grading System"
                )
            else:
                mlflow.keras.save.log_model(
                    model = model, 
                    artifact_path = "Model_info", 
                    signature = infer_signature(X_test, model.predict(X_test)),
                    registered_model_name = "ESNE"  #Event severity Network estimator
                    )

    mlflow.end_run()
    return None