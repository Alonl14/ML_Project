from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ParticleDataset(Dataset):
    def __init__(self, data_path, norm_path, QT):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data[' pdg'].isin([2112])]  # 22 - photons , 2112 - neutrons
        self.data[' rx'] = np.sqrt(self.data[' xx'].values ** 2 + self.data[' yy'].values ** 2)
        self.data[' rp'] = np.sqrt(self.data[' pxx'].values ** 2 + self.data[' pyy'].values ** 2)
        self.data[' phi_p'] = np.arctan2(self.data[' pyy'].values, self.data[' pxx'].values) + np.pi
        self.data[' phi_x'] = np.arctan2(self.data[' yy'].values, self.data[' xx'].values) + np.pi
        self.data = self.data[
            [" rx", " xx", " yy", " rp", " phi_p", " pzz", " eneg", " time"]]  # ,' xx',' yy',' pxx',' pyy'
        self.preprocess = self.data.values
        self.norm = pd.read_csv(norm_path, index_col=0)
        for col in self.norm.index:
            self.data[col] = (self.data[col] - self.norm['min'][col] + 10 ** (-15)) / (
                        self.norm['max'][col] - self.norm['min'][col] + 2 * 10 ** (-15))
        self.data[' pzz'] = 1 - self.data[' pzz']
        self.data[[' rp', ' pzz', ' eneg', ' time']] = -np.log(self.data[[' rp', ' pzz', ' eneg', ' time']])  #
        self.data[' yy'] = np.log(self.data[' yy'] / self.data[' rx'])
        self.data[' xx'] = np.log(self.data[' xx'] / self.data[' rx'])
        self.data[' pzz'] = 1 / (self.data[' pzz'] ** 2 / 15 + 1)
        self.data[' rx'] = np.sqrt(self.data[' rx'])

        self.preqt = self.data.values
        self.quantiles = QT.fit(self.data)
        self.data = QT.fit_transform(self.data)
        self.data = self.data.astype(np.float32)

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]