import numpy as np
import keras
import imageio

def prepare_data(X: np.ndarray) -> np.ndarray:
    """ Pad a 28x28 picture into a 32x32 picture """
    X_new = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return X_new

def parse_file(folder: str, path: str, labels: dict) -> list:
    ids = []

    for line in open(path, "r"):
        label, _ = line.strip().split("/")
        img_full_path = "data/" + folder + "/" + line.strip()
        ids.append(img_full_path)
        labels[img_full_path] = int(label)

    return ids

def generate_generator_objects() -> tuple:
    labels = { }
    X_train_ids = parse_file("train", "train_images_paths", labels)
    X_val_ids = parse_file("val", "val_images_paths", labels)
    X_test_ids = parse_file("test", "test_images_paths", labels)

    return ({ "train": X_train_ids, "validation": X_val_ids, "test": X_test_ids }, labels)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = np.array([imageio.imread(ID)]).T
            img = prepare_data(np.array([img]))
            X[i,] = img[0]

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)