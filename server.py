import base64
import numpy as np
import pandas as pd
from PIL import Image
from feature_extractor import FeatureExtractor
from flask import Flask, request, render_template


app = Flask(__name__)

# Load features extractors
fe = FeatureExtractor()
# Load clusters centroids
centroids = np.load(
    r"C:\Users\DELL\Downloads\sis-master\sis-master\centroids.pkl", allow_pickle=True)
# Load cluster_image matching file
f = pd.read_csv(
    r"C:\Users\DELL\Downloads\sis-master\sis-master\result_cluster.csv")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        
        query_img_64 = base64.b64encode(file.read()).decode('utf-8')
        img = Image.open(file.stream)  # PIL image

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(centroids-query, axis=1)
        ids = np.argsort(dists)[0:2]   # Cluster number
        
        # Images features based on the first centroid
        features = np.load(
            "C:/Users/DELL/Downloads/sis-master/sis-master/static/results/features/" + str(ids[0]) + ".npy", allow_pickle=True)
        # Images paths based on the first centroid
        img_paths = np.load(
            "C:/Users/DELL/Downloads/sis-master/sis-master/static/results/imgfile/" + str(ids[0]) + ".npy", allow_pickle=True)
       
        # L2 distances to features
        distances = np.linalg.norm(features-query, axis=1)

        # Images features based on the second centroid
        features = np.load(
            "C:/Users/DELL/Downloads/sis-master/sis-master/static/results/features/" + str(ids[1]) + ".npy", allow_pickle=True)
        
        # Images paths based on the second centroid
        img_paths = np.append(img_paths, np.load(
            "C:/Users/DELL/Downloads/sis-master/sis-master/static/results/imgfile/" + str(ids[1]) + ".npy", allow_pickle=True)
        )
        # Update the distances
        distances = np.append(
            distances, np.linalg.norm(features-query, axis=1))
        
        # Pick up the 30 most relevant images
        ids_image = np.argsort(distances)[:30]  # Top 30 results

        scores = [(distances[id], img_paths[id]) for id in ids_image]

        return render_template('index.html',
                               query_path=query_img_64,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("localhost")
