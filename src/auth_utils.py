import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import os

# Paths for persistent user database
EMBEDDINGS_FILE = "registered_users.npy"
LABELS_FILE = "user_labels.npy"

# Load or initialize database
if os.path.exists(EMBEDDINGS_FILE):
    registered_embeddings = np.load(EMBEDDINGS_FILE)
    user_labels = np.load(LABELS_FILE)
else:
    registered_embeddings = np.empty((0, 128))  # 128 = embedding size
    user_labels = np.array([])

# Load trained model
embedding_model = load_model("ecg_auth_model.h5")

# Get embedding-only model (no softmax)
if embedding_model.layers[-1].activation.__name__ == 'softmax':
    from tensorflow.keras import Model
    embedding_only_model = Model(
        inputs=embedding_model.input,
        outputs=embedding_model.layers[-2].output
    )
else:
    embedding_only_model = embedding_model


def extract_embedding(ecg_sample):
    """Convert raw ECG sample to embedding."""
    sample = np.expand_dims(ecg_sample, axis=0)  # (1, timesteps, 1)
    embedding = embedding_only_model.predict(sample, verbose=0)
    return embedding[0]


def register_user(username, ecg_samples):
    """
    Register a new user with one or more ECG samples.
    ecg_samples: list or numpy array of shape (n_samples, timesteps, 1)
    """
    global registered_embeddings, user_labels

    # Ensure it's iterable
    if isinstance(ecg_samples, np.ndarray) and ecg_samples.ndim == 2:
        ecg_samples = [ecg_samples]  # single sample

    for sample in ecg_samples:
        embedding = extract_embedding(sample)
        registered_embeddings = np.vstack([registered_embeddings, embedding])
        user_labels = np.append(user_labels, username)

    # Save database
    np.save(EMBEDDINGS_FILE, registered_embeddings)
    np.save(LABELS_FILE, user_labels)
    return True


def authenticate_user(ecg_sample, threshold=0.85):
    """Authenticate a user against registered database."""
    if len(user_labels) == 0:
        return "No users registered", 0.0

    user_embedding = extract_embedding(ecg_sample)
    similarities = cosine_similarity([user_embedding], registered_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]

    if max_sim >= threshold:
        return user_labels[max_idx], max_sim
    else:
        return "Unknown", max_sim
