from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np

kmeans = KMeans(200)

features = []
img_paths = []
for feature_path in Path(r"C:\Users\DELL\Downloads\sis-master\sis-master\static\feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path(r"C:\Users\DELL\Downloads\sis-master\sis-master\static\img") / (feature_path.stem + ".jpg"))
features = np.array(features)

kmeans.fit(features)
y = kmeans.fit_predict(features)

import pandas as pd
df = pd.DataFrame({'col1':img_paths, 'col2':y})
df.to_csv(r"C:\Users\DELL\Downloads\sis-master\sis-master\result.csv", header=False, index=False)

centroids = kmeans.cluster_centers_

from feature_extractor import FeatureExtractor
from PIL import Image
fe = FeatureExtractor()
feature = fe.extract(img = Image.open(r"C:\Users\DELL\Downloads\195897274.jpg"))

distance = np.linalg.norm(centroids - feature, axis= 1)

print(distance)
