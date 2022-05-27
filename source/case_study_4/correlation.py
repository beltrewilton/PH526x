import pandas as pd
import matplotlib.pyplot as plt

regioons = pd.read_csv('./regions.txt')
whiskies = pd.read_csv('./whiskies.txt')

flavors = whiskies.iloc[:, 2:14]
# corr_flavors = pd.DataFrame.corr(flavors)
# plt.figure(figsize=(10, 8))
# plt.pcolor(corr_flavors)
# plt.colorbar()
# plt.show()

corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10, 8))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.show()
