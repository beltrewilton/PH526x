import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster._bicluster import SpectralCoclustering

regioons = pd.read_csv('./regions.txt')
whiskies = pd.read_csv('./whiskies.txt')
flavors = whiskies.iloc[:, 2:14]
corr_whisky = pd.DataFrame.corr(flavors.transpose())

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)

# extract the group labels from the model
whiskies['Group'] = pd.Series(model.row_labels_, index=whiskies.index)

# specify their index explicitly
whiskies = whiskies.loc[np.argsort(model.row_labels_)]

# reorder the rows in increasing order by group labels
whiskies = whiskies.reset_index(drop=True)

# to numpy
correlations = pd.DataFrame.corr(whiskies.iloc[:, 2:14].transpose())
correlations = np.array(correlations)
