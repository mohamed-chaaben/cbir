from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*/*.jpg")):
        feature = fe.extract(img=Image.open(img_path))
        img_path_str = str(img_path)
        feature_path = Path("./static/feature") / (img_path_str[11:11+img_path_str[11:].index('/')] + '-' + img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
        print(feature_path)