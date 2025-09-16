import argparse
import tensorflow as tf
from utils.data_loader import load_data
from models.aeformer import build_aeformer

def main(args):
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data)

    # Build model
    model = build_aeformer(input_shape=(1000, 1), num_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, lr_schedule]
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Final Test Accuracy: {test_accuracy*100:.2f}% | Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AEFormer on AE dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to .mat dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    main(args)
