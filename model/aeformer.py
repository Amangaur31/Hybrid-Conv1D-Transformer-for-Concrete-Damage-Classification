import tensorflow as tf

# Transformer Encoder Block
def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.5):
    # Multi-head attention
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ff = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(inputs.shape[-1])
    ])
    x = ff(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# AEFormer Model
def build_aeformer(input_shape=(1000, 1), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    # Conv1D front-end
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Transformer encoder
    x = transformer_encoder(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="AEFormer")
    return model
