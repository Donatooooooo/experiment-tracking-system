from data.dataset import Dataset
from models.randomForest import RandomForestTrainer
from models.MLP import MLPTrainer
from MLFlowTracker import trainAndLog
from utils.preprocessor import preprocessing
from utils.utility import drop_bestparams
import dagshub

drop_bestparams(True)

dagshub.init(repo_owner='donatooooooo', repo_name='MLflow_Server', mlflow=True)

experimentName = "Cancer_classification"
dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessing(dataset, cluster = False)
trainer = RandomForestTrainer('diagnosis', ['diagnosis'], dataset)

trainAndLog(
    dataset = dataset,
    trainer = trainer,
    experimentName = experimentName,
    datasetName = "Breast_Cancer_Wisconsin.csv",
    model_name = "Histological_Grading_System"
)

experimentName = "Cancer_area_prediction"
dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessing(dataset, cluster = False)
trainer = MLPTrainer('area_mean', ['area_mean'], dataset)

trainAndLog(
    dataset = dataset,
    trainer = trainer,
    experimentName = experimentName,
    datasetName = "Breast_Cancer_Wisconsin.csv",
    model_name = "Histological_Grading_System"
)