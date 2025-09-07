import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Add, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Lambda, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.activations import gelu


# ------------------ config / seeds ------------------
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "6975118"
JET_FILES = {"g.hdf5": 0, "q.hdf5": 1, "t.hdf5": 2, "w.hdf5": 3, "z.hdf5": 4}


# ------------------ data loading ------------------
def load_jetnet(data_dir: str, jet_files: dict):
    xs, ys = [], []
    for fname, label in jet_files.items():
        with h5py.File(os.path.join(data_dir, fname), "r") as f:
            feats = f["particle_features"][:]  # (n, 30, 4)
            xs.append(feats)
            ys.append(np.full((feats.shape[0],), label, dtype=np.int32))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


# ------------------ model components ------------------
def transformer_block(x, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1):
    # pre-norm self-attention
    h = LayerNormalization()(x)
    h = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(h, h)
    h = Dropout(dropout_rate)(h)
    x = Add()([x, h])

    # pre-norm feed-forward
    h = LayerNormalization()(x)
    h = Dense(ff_dim, activation=gelu)(h)
    h = Dense(x.shape[-1])(h)
    h = Dropout(dropout_rate)(h)
    x = Add()([x, h])
    return x


class PMASeeds(tf.keras.layers.Layer):
    """Pooling by Multi-Head Attention with k learnable seeds."""
    def __init__(self, num_seeds=4, num_heads=4, key_dim=64):
        super().__init__()
        self.k = num_seeds
        self.key_dim = key_dim
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def build(self, input_shape):
        self.seeds = self.add_weight(
            shape=(1, self.k, self.key_dim),
            initializer="random_normal",
            trainable=True,
            name="pma_seeds",
        )

    def call(self, x):
        b = tf.shape(x)[0]
        seeds = tf.tile(self.seeds, [b, 1, 1])       # (b, k, d)
        pooled = self.attn(seeds, x)                 # (b, k, d)
        return tf.reshape(pooled, [b, self.k * self.key_dim])


def se_channel_attention(x, reduction=8):
    c = x.shape[-1]
    g = GlobalAveragePooling1D()(x)                  # (b, c)
    h = Dense(max(c // reduction, 1), activation="relu")(g)
    s = Dense(c, activation="sigmoid")(h)
    s = Lambda(lambda t: tf.expand_dims(t, 1))(s)    # (b, 1, c)
    return Multiply()([x, s])


def build_model():
    inputs = Input(shape=(30, 4), name="jet_input")
    x = Dense(64, activation=gelu)(inputs)

    # 3 transformer encoder blocks
    for _ in range(3):
        x = transformer_block(x, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1)

    # channel (SE) attention + PMA with 4 seeds
    x = se_channel_attention(x)
    x = PMASeeds(num_seeds=4, num_heads=4, key_dim=64)(x)

    # classifier head
    x = Dense(256, activation=gelu)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation=gelu)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(5, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------ main ------------------
def main():
    X_raw, y = load_jetnet(DATA_DIR, JET_FILES)
    print("data:", X_raw.shape, "classes:", np.unique(y))

    # normalize per-particle features
    flat = X_raw.reshape(-1, 4)
    scaler = StandardScaler()
    flat = scaler.fit_transform(flat)
    X = flat.reshape(-1, 30, 4)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("splits:", X_train.shape, X_val.shape)

    model = build_model()
    model.summary()

    # callbacks
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ckpt = ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=1024,
        callbacks=[es, rlrop, ckpt],
        verbose=1,
    )

    model.save("jetnet_model.keras")

    # training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Figure_1.png", dpi=150)
    plt.show()

    # evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("Figure_2.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
