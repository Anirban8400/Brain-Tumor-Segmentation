import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import pandas as pd

from unet import build_unet
from metrics import dice_loss, dice_coeff
from train import load_dataset

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    
    mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (H, W, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  # (H, W, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  # (H, W, 3)
    y_pred = (y_pred * 255)  # convert to uint8
    line=np.ones((H,10,3))*255
    cat_images = np.concatenate([image,line, mask,line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    # if not success:
    #     print(f"Failed to save image: {save_image_path}")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("results")

    """Load Model"""
    with CustomObjectScope({"dice_coeff": dice_coeff, "dice_loss": dice_loss}): #dictionary
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))

    """Load Dataset"""
    dataset_path = r"C:\Users\ATLANTIS\Downloads\archive"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    """Evaluation"""
    score = []
    for x,y in tqdm(zip(test_x, test_y), total=len(test_y)):
        name = os.path.basename(x) # or os.path.basename(img_path)
        # Load and preprocess image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        # Load and preprocess mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        # Predict
        y_pred = model.predict(x, verbose=0)[0]
        
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred=y_pred>=0.5
        y_pred=y_pred.astype(np.int32)
        
        # Save image
        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)

    #     # Metrics
        mask_bin=mask/255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred_flat = y_pred.flatten()

        f1 = f1_score(mask, y_pred_flat, average='binary')
        jaccard = jaccard_score(mask, y_pred_flat, average='binary')
        recall_val = recall_score(mask, y_pred_flat, average='binary')
        precision_val = precision_score(mask, y_pred_flat, average='binary')
        score.append([name, f1, jaccard, recall_val, precision_val])
    score =[s[1:] for s in score]
    score=np.mean(score, axis=0)
    print(f"F1: {score[0]:0.5f}")
    print(f"Jaccard: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    """Save CSV and print averages"""
    df = pd.DataFrame(score, columns=["IMAGE", "F1", "JACCARD", "RECALL", "PRECISION"])
    df.to_csv("files/score.csv")
