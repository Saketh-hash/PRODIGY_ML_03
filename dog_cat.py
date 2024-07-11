import os 
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor
import logging
import psutil
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
extrpath = r'D:\Saketh\PROJECTS\dogs-vs-cats'  #Add your path from your LOCAL SYSTEM of the dataset.
trainpath = os.path.join(extrpath, 'train')
testpath = os.path.join(extrpath, 'test1')
logging.info(f"Train path exists: {os.path.exists(trainpath)}")
logging.info(f"Test path exists: {os.path.exists(testpath)}")
imgsize = 64
btchsize = 5000
def preprocess_img(imgpath):
    try:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (imgsize, imgsize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.flatten()
        return img
    except Exception as e:
        logging.error(f"Error processing Image {imgpath}: {e}")
        return None
def loadimginbtches(imgpaths, label, btchsize):
    X, y = [], []
    for i in range(0, len(imgpaths), btchsize):
        btchpaths = imgpaths[i:i+btchsize]
        with ThreadPoolExecutor() as executor:
            results = executor.map(preprocess_img, btchpaths)
            for img in results:
                if img is not None:
                    X.append(img)
                    y.append(label)
        logging.info(f"Processed batch {i // btchsize + 1}/{len(imgpaths)//btchsize + 1}")
    return X, y
catimgs = [os.path.join(trainpath, img) for img in os.listdir(trainpath) if 'cat' in img]
dogimgs = [os.path.join(trainpath, img) for img in os.listdir(trainpath) if 'dog' in img]
logging.info(f"Number of cat images: {len(catimgs)}")
logging.info(f"Number of dog images: {len(dogimgs)}")
Xcat, ycat = loadimginbtches(catimgs, 0, btchsize)
Xdog, ydog = loadimginbtches(dogimgs, 1, btchsize)
X = np.array(Xcat + Xdog)
y = np.array(ycat + ydog)
logging.info(f"Shape of X: {X.shape}")
logging.info(f"Shape of y: {y.shape}")
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Dataset is empty. Check the paths and dataset contents.")
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state=42)
svmmodel = SVC(kernel= 'linear', random_state=42)
svmmodel.fit(Xtrain, ytrain)
ypredict = svmmodel.predict(Xtest)
accur = accuracy_score(ytest, ypredict)
logging.info(f"Accuracy --> {accur}")
logging.info(f"Classification report: \n{classification_report(ytest, ypredict)}")
logging.info(f"Memory usage --> {psutil.virtual_memory().percent}%")