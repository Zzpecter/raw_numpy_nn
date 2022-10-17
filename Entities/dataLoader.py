import idx2numpy
import numpy as np


class DataLoader:

    @staticmethod
    def load_data(dataset='training'):

        if dataset == 'training':
            images_file = 'Data/Training/train-images.idx3-ubyte'
            labels_file = 'Data/Training/train-labels.idx1-ubyte'
        elif dataset == 'test':
            images_file = 'Data/Training/train-images.idx3-ubyte'
            labels_file = 'Data/Training/train-labels.idx1-ubyte'
        else:
            raise ValueError(f"Error! Dataset: {dataset} is not recognized as valid. Use 'training' or 'test'")

        data = idx2numpy.convert_from_file(images_file)
        labels = idx2numpy.convert_from_file(labels_file)

        return data, labels
