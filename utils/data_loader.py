import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def load_data(file_path, num_classes=3, test_size=0.3, val_size=0.5):
    """
    Load and preprocess Acoustic Emission data from .mat file.
    
    Args:
        file_path (str): Path to .mat dataset
        num_classes (int): Number of output classes
        test_size (float): Fraction of data for test+val
        val_size (float): Fraction of temp data for validation split

    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    mat = loadmat(file_path)

    X = mat['x_train']  
    y_raw = mat['y_train'].flatten()

    # Convert labels if they are strings
    if y_raw.dtype == 'O':
        y = np.array([str(label[0]) for label in y_raw])
    else:
        y = y_raw

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y, num_classes=num_classes)

    # Normalize signals
    X = X / np.max(np.abs(X))
    X = X[..., np.newaxis]  # add channel dim for Conv1D

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y.argmax(axis=1), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp.argmax(axis=1), random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
