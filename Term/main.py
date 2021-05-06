import pandas as pd
import numpy as np
from matplotlib import cm
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = []
    xcoords = np.linspace(250, 400, 76)
    ycoords = np.linspace(250, 600, 351)
    samplesZcoords = []
    for i in range(1, 18):
        with open(str(i)+'.txt') as f:
            content = list(filter(None, f.read().split('\n')))
            del content[0: 4]
            sample = []
            Zcoords = []
            for line in content:
                line = line.split('\t')
                del line[0]
                zRow = []
                for l in line:
                    k = float(l.replace(',', '.'))
                    if i < 3 and k >= 300:
                        k = 300
                    if i >= 3 and k >= 100:
                        k = 100
                    sample.append(k)
                    zRow.append(k)
                Zcoords.append(zRow)
            Zcoords = median_filter(Zcoords, footprint=np.ones((5, 5)), mode='constant')
            samplesZcoords.append(Zcoords)
            data.append(sample)
    datak = np.array(data).T.tolist()

    grids = np.meshgrid(xcoords, ycoords)
    for i in range(17):
        cs = plt.contourf(xcoords, ycoords, samplesZcoords[i], levels=20, cmap=cm.tab20b)
        plt.colorbar(cs)
        plt.title(str(i) + ' Sample')
        plt.savefig(str(i) + '.png')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    scaled_dta = preprocessing.scale(datak)
    pca = PCA()
    pca.fit(scaled_dta)
    pca_data = pca.transform(scaled_dta)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    pca_df = pd.DataFrame(pca_data, columns=labels)
    plt.savefig('Histogram.png')
    plt.show()

    PC1_Grid = np.reshape(pca_df.PC1.to_numpy(), (351, 76))
    PC2_Grid = np.reshape(pca_df.PC2.to_numpy(), (351, 76))
    xcoords = xcoords.tolist()
    ycoords = ycoords.tolist()

    z1 = median_filter(PC1_Grid, footprint=np.ones((5, 5)), mode='constant')
    z2 = median_filter(PC2_Grid, footprint=np.ones((5, 5)), mode='constant')
    cs = plt.contourf(xcoords, ycoords, z1, levels=20, cmap=cm.tab20b)
    plt.colorbar(cs)
    plt.title('First Principal Component')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('PC1.png')
    plt.show()
    cs = plt.contourf(xcoords, ycoords, z2, levels=20, cmap=cm.tab20b)
    plt.colorbar(cs)
    plt.title('Second Principal Component')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('PC2.png')
    plt.show()

    PCs = []
    PCs.append(pca_df.PC1.to_numpy())
    PCs.append(pca_df.PC2.to_numpy())
    projection = scaled_dta.T.dot(np.array(PCs).T)
    plt.scatter(projection[:,0], projection[:,1])
    plt.show()





