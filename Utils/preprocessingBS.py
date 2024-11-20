from Dataset.dataset import Dataset
from Utils.kmeans import kMeans
import pandas, re

def browser(userAgent):
    browserPattern = re.compile(r'Mozilla.*?(Firefox|Chrome|MSIE|Safari)', re.IGNORECASE)
    browserMatch = browserPattern.search(userAgent)
    if browserMatch:
        return browserMatch.group(1)
    elif userAgent.startswith("Opera"):
        return "Opera"

def os(userAgent):
    osPattern = re.compile(r'(Windows|Mac OS|Linux|iPhone OS|iPad OS|iPod OS|Android)', re.IGNORECASE)
    osMatch = osPattern.search(userAgent)
    if osMatch:
        return osMatch.group(1)
    return None

def basicPreprocessing(dataset: Dataset):
    data = dataset.getDataset()
    data['Proxy Information'] = data['Proxy Information'].apply(lambda x: 1 if pandas.notna(x) else 0)
    data['Browser'] = data['Device Information'].apply(lambda x: browser(x) if pandas.notnull(x) else None)
    data['OS'] = data['Device Information'].apply(lambda x: os(x) if pandas.notnull(x) else None)
    dataset.setDataset(data)
    dataset.dropDatasetColumns(['Source IP Address', 'Timestamp', 'Destination IP Address', 
                                    'Payload Data', 'Attack Signature', 'User Information', 'Severity Level', 
                                        'Network Segment', 'Geo-location Data', 'Device Information'])
    return dataset

def emptyValues(dataset: Dataset):
    dataset.emptyValues('Alerts/Warnings', 'Alert Triggered')
    dataset.emptyValues('Malware Indicators', 'IoC Detected')
    dataset.emptyValues('Firewall Logs', 'Log Data')
    dataset.emptyValues('IDS/IPS Alerts', 'Alert Data')
    return dataset

def getDummies(dataset: Dataset):
    dataset.getDummies('Packet Type')
    dataset.getDummies('Action Taken')
    dataset.getDummies('Attack Type')
    dataset.getDummies('Traffic Type')
    dataset.getDummies('Log Source')
    dataset.getDummies('OS')
    dataset.getDummies('Browser')
    dataset.getDummies('Protocol')
    return dataset

def normalizeColumns(dataset: Dataset):
    dataset.normalizeColumn('Source Port')
    dataset.normalizeColumn('Destination Port')
    dataset.normalizeColumn('Packet Length')
    dataset.normalizeColumn('Anomaly Scores')
    dataset.normalizeColumn('Basescore')
    return dataset

def preprocessingBS(dataset: Dataset, cluster = False):
    dataset = basicPreprocessing(dataset)
    dataset = emptyValues(dataset)
    dataset = normalizeColumns(dataset)
    dataset = getDummies(dataset)
    dataset.replaceBoolean()
    
    if cluster:
        features = dataset.getDataFrame(['Source Port','Destination Port','Packet Length'])
        kmeans = kMeans().clustering(features)
        dataset.addDatasetColumn('Network Features Cluster', kmeans.fit_predict(features))
        dataset.dropDatasetColumns(columnsToRemove=['Source Port','Destination Port','Packet Length'])
        dataset.normalizeColumn('Network Features Cluster')
    return dataset