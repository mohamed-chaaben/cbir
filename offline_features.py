# Offline extarcting features

import numpy as np
import pandas as pd

f = pd.read_csv(
    r"C:\Users\DELL\Downloads\sis-master\sis-master\result_cluster.csv")


for ids in range(0, 200): # We have 200 clusters
    features = []
    img_path = []
    for i in range(5000*ids, 5000*ids+5000):
        path_feature = f.iloc[i]['name']
        features.append(np.load(
            "C:/Users/DELL/Downloads/sis-master/sis-master/static/feature/"+path_feature+".npy"))
        s = path_feature.index("-")
        img = path_feature[:s]+"/"+path_feature[s+1:]
        img_path.append("img" + "/" + img + ".jpg")
    np.save("C:/Users/DELL/Downloads/sis-master/sis-master/static/results/features/" +
            str(ids) + ".npy", np.array(features))
    np.save("C:/Users/DELL/Downloads/sis-master/sis-master/static/results/imgfile/" +
            str(ids) + ".npy", np.array(img_path))
