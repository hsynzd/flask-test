import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        self.classes_ = np.array(sorted(set(y)))
        self.classes_ = np.append(self.classes_, ['UNK'])  # Add a class for unknown labels
        return self

    def transform(self, y):
        unknown_label = self.classes_.tolist().index('UNK')
        return np.array([self.classes_.tolist().index(x) if x in self.classes_ else unknown_label for x in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return np.array([self.classes_[i] if i < len(self.classes_) else 'UNK' for i in y])
