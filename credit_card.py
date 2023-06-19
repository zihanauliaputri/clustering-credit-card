import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('credit.csv')

X = df.drop(['Sl_No','Customer Key'], axis=1)

st.header("isi dataset")
st.write(df)

clusters=[]
for i in range(1,11):
  km = KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clus = st.sidebar.slider("Pilih jumlah cluster: ", 2,10,2,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=X, x='Total_Credit_Cards', y='Avg_Credit_Limit', hue='Labels', markers=True, size='Labels', palette=sns.color_palette('hls', n_clust))

    for label in X['Labels'].unique():
        plt.annotate(label,
                    (X[X['Labels'] == label]['Total_Credit_Cards'].mean(),
                      X[X['Labels'] == label]['Avg_Credit_Limit'].mean()),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=20, weight='bold',
                    color='black')
    
    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)

k_means(clus)
    