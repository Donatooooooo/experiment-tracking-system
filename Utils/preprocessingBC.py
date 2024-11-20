from Dataset.dataset import Dataset
from Utils.kmeans import kMeans

def preprocessingBC(dataset : Dataset, cluster = False):
    dataset.dropDatasetColumns(["id"])
    dataset.replaceBoolean("M", "B")
    for column in dataset.getDataset().columns:
        dataset.normalizeColumn(column)
    
    if cluster:
        features = dataset.getDataFrame(['radius_mean', 'texture_mean', 'perimeter_mean'])
        kmeans = kMeans().clustering(features)
        dataset.addDatasetColumn('Appearance Cluster', kmeans.fit_predict(features))
        dataset.dropDatasetColumns(columnsToRemove=['radius_mean', 'texture_mean', 'perimeter_mean'])
        dataset.normalizeColumn('Appearance Cluster')
        
    dataset.dropDatasetColumns(['Unnamed: 32'])
    return dataset