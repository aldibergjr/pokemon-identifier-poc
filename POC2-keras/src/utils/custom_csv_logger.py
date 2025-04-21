import csv
import tensorflow as tf

class CustomCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.file = open(self.filename, 'w', newline='', encoding='utf-8')
        self.writer = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=['epoch'] + list(logs.keys()))
            self.writer.writeheader()
        row = {'epoch': epoch + 1}
        row.update(logs)
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, logs=None):
        self.file.close()
