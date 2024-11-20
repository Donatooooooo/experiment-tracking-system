from Dataset.dataset import Dataset
from Models.randomForest import RandomForestTrainer
from Models.MLP import MLPTrainer
from MLFlowTracker import trainAndLog
from Utils.preprocessingBC import preprocessingBC
from Utils.preprocessingBS import preprocessingBS
from Utils.utility import drop_bestparams


drop_bestparams(True)

experimentName = "Cancer_classification"
dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessingBC(dataset, cluster = False)
trainer = RandomForestTrainer('diagnosis', ['diagnosis'], dataset)

trainAndLog(
    dataset = dataset,
    trainer = trainer,
    experimentName = experimentName,
    datasetName = "Breast_Cancer_Wisconsin.csv",
    signature = 0,
    tags = {"Training Info": "missing info"}
)


experimentName = "Basescore_prediction"
dataset = Dataset("Dataset/network_events.csv")
dataset = preprocessingBS(dataset, cluster = False)
trainer = MLPTrainer('Basescore', ['Basescore'], dataset)

trainAndLog(
    dataset = dataset,
    trainer = trainer,
    experimentName = experimentName,
    datasetName = "Cyber_Security_Attacks.csv",
    signature = 1,
    tags = {"Training Info": "missing info"}
)
