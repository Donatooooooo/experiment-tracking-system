import mlflow, numpy as np, pandas as pd, os, shutil
from Dataset.dataset import Dataset

def inference(dataset : Dataset, modelInfo, X_test, y_test):
    loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
    predictions = loadedModel.predict(X_test)
    featureNames = dataset.getDataset().columns.tolist()
    result = pd.DataFrame(X_test, columns = featureNames)
    
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result.sample(100).to_csv('ModelTracker/Utils/predictions.csv', index=False)
    
def drop_bestparams(do = False):
    if do:
        folder = "Utils/best_params"
        if os.path.exists(folder) and os.path.isdir(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)