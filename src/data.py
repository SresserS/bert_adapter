import pandas as pd
from torch.utils.data import Dataset
import csv

class DataFrameTextClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 x_label: str = 'content',
                 y_label: str = 'label'):
        self.x = df[x_label]
        self.length = len(self.x)

        self.y = df[y_label].astype('category')
        self.n_classes = len(self.y.cat.categories)
        self.y = self.y.cat.codes

    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        return {'x': str(x), 'y': int(y)}

    def __len__(self):
        return self.length

    @staticmethod
    def from_file(file_path: str,
                  x_label: str = 'sentence',
                  y_label: str = 'label'):
        df = pd.read_csv(file_path)
        return DataFrameTextClassificationDataset(df, x_label, y_label)

class DataIOSST2(object):
    def __init__(self, config):
        self.path = config['path']
        self.batch_size = config['batch_size']
        self.train_word, self.train_label, \
        self.dev_word, self.dev_label, \
        self.test_word, \
        self.test_label = self.read_train_dev_test()

    def read_train_dev_test(self):
        train_word, train_label = self.read_data(self.path + '/train.tsv')
        dev_word, dev_label = self.read_data(self.path + '/dev.tsv')
        test_word, test_label = self.read_data(self.path + '/test.tsv')
        return train_word, train_label, dev_word, dev_label, test_word, test_label

    @staticmethod
    def read_data(path):
        data = []
        label = []
        csv.register_dialect('my', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(path) as tsvfile:
            file_list = csv.reader(tsvfile, "my")
            first = True
            for line in file_list:
                if first:
                    first = False
                    continue
                data.append(line[1])
                label.append(int(line[0]))
        csv.unregister_dialect('my')
        return data, label


class SSTDataset(Dataset):
    def __init__(self, sentences, labels):
        self.dataset = sentences
        self.labels = labels

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)
