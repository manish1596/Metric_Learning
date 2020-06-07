import numpy as np
import pandas as pd
from numpy import linalg as LA
from metric_learn import LMNN
from metric_learn import NCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.datasets import load_iris
iris_data = load_iris()

def get_cluster_data(n_clusters, n_points_per_cluster, mean_radius, feature_dim):
    cluster_centers = np.random.randn(n_clusters, feature_dim)
    cluster_centers = np.random.uniform(low=0.99*mean_radius, high=mean_radius, size=(n_clusters, 1)) * cluster_centers / np.expand_dims(np.linalg.norm(cluster_centers, axis = 1), axis=1)
    X = np.zeros((n_clusters*n_points_per_cluster, feature_dim))
    Y = np.zeros(n_clusters*n_points_per_cluster, dtype=np.int64)
    for i in range(n_clusters):
        random_mat = np.random.uniform(low=0.0, high=1.0, size=(feature_dim, feature_dim))
        Q = np.dot(random_mat, random_mat.T)
        w,u=LA.eig(Q)
        cov_skewed = np.array([np.random.uniform(2*(i+5),2*i+11) for i in range(feature_dim)])
        #cov_skewed=np.random.permutation(cov_skewed)
        #cov_mat = np.diag(cov_skewed)
        cov_mat = np.dot(np.dot(u, np.diag(cov_skewed)), u.T)
        #for f in range(feature_dim):
            #random_mat[f][f] *= 10
        #print(LA.eig(cov_mat))
        X[i*n_points_per_cluster:(i+1)*n_points_per_cluster] = np.random.multivariate_normal(mean=cluster_centers[i], cov=cov_mat, size=(n_points_per_cluster))
        Y[i*n_points_per_cluster:(i+1)*n_points_per_cluster] = i
    perm = np.random.permutation(n_clusters*n_points_per_cluster)
    X = X[perm]
    Y = Y[perm]
    return X, Y

def lmnn_fit(X_train, Y_train, X_test, Y_test, color_map):
    lmnn = LMNN(init='pca', k=3, learn_rate=5e-4, max_iter=500000, regularization=0.2)
    lmnn.fit(X_train, Y_train)
    X_train_transformed = lmnn.transform(X_train)
    if(X_train.shape[1]==2):
        plt.figure()
        plt.scatter(X_train_transformed[:,0], X_train_transformed[:,1], c=color_map[Y_train], s=2)
        plt.savefig("after_lmnn_transform_train.png", dpi=300)
    X_test_transformed = lmnn.transform(X_test)
    if(X_test.shape[1]==2):
        plt.figure()
        plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=color_map[Y_test], s=2)
        plt.savefig("after_lmnn_transform_test.png", dpi=300)
    return (X_train_transformed, X_test_transformed)

def nca_fit(X_train, Y_train, X_test, Y_test, color_map):
    nca = NCA(init='pca', max_iter=5000)
    nca.fit(X_train, Y_train)
    X_train_transformed = nca.transform(X_train)
    if(X_train.shape[1]==2):
        plt.figure()
        plt.scatter(X_train_transformed[:,0], X_train_transformed[:,1], c=color_map[Y_train], s=2)
        plt.savefig("after_nca_transform_train.png", dpi=300)
    X_test_transformed = nca.transform(X_test)
    if(X_test.shape[1]==2):
        plt.figure()
        plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=color_map[Y_test], s=2)
        plt.savefig("after_nca_transform_test.png", dpi=300)
    return (X_train_transformed, X_test_transformed)

def experiment(r, num_clusters, n_points_per_cluster, feature_dim, model):
    color_map = cm.rainbow(np.linspace(0,1,num_clusters))
    X, Y = get_cluster_data(num_clusters, n_points_per_cluster, r, feature_dim)

    train_size = int(0.8 * X.shape[0])
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    if(X_train.shape[1] == 2):
        plt.figure()
        plt.scatter(X_train[:,0], X_train[:,1], c=color_map[Y_train], s=2)
        #plt.savefig("before_transform_train.png", dpi=300)
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    if(X_test.shape[1] == 2):
        plt.figure()
        plt.scatter(X_test[:,0], X_test[:,1], c=color_map[Y_test], s=2)
        #plt.savefig("before_transform_test.png", dpi=300)
    knn1 = KNeighborsClassifier(n_neighbors=10)
    knn1.fit(X_train, Y_train)
    Y_pred = knn1.predict(X_test)
    knn_accuracy_before = float(np.sum(Y_pred==Y_test))/len(Y_test)
    #print("KNN accuracy before transformation = {}".format(float(np.sum(Y_pred==Y_test))/len(Y_test)))

    if(model == 'lmnn'):
        X_train_transformed, X_test_transformed = lmnn_fit(X_train, Y_train, X_test, Y_test, color_map)
    if(model == 'nca'):
        X_train_transformed, X_test_transformed = nca_fit(X_train, Y_train, X_test, Y_test, color_map)
    knn2 = KNeighborsClassifier(n_neighbors=10)
    knn2.fit(X_train_transformed, Y_train)
    Y_pred = knn2.predict(X_test_transformed)
    knn_accuracy_after = float(np.sum(Y_pred==Y_test))/len(Y_test)
    #print("KNN accuracy after transformation = {}".format(float(np.sum(Y_pred==Y_test))/len(Y_test)))

    return [knn_accuracy_before, knn_accuracy_after]

n_exp = 1
df = pd.DataFrame(columns = ['10-NN accuracy','radius','type'])
for e in range(n_exp):
    print("Experiment " + str(e) + " done")
    before_acc = []
    after_acc = []
    for r in range(10,11):
        num_clusters = 8
        n_points_per_cluster = 200
        feature_dim = 2
        knn_accuracy_before, knn_accuracy_after = experiment(r, num_clusters, n_points_per_cluster, feature_dim, model='nca')
        before_acc.append(knn_accuracy_before)
        after_acc.append(knn_accuracy_after)

    df = pd.concat([df, pd.DataFrame({'10-NN accuracy':np.array(before_acc), 'radius':np.arange(len(before_acc)), 'type':'Before'})])
    df = pd.concat([df, pd.DataFrame({'10-NN accuracy':np.array(after_acc), 'radius':np.arange(len(after_acc)), 'type':'After'})])
#sns.relplot(x='radius', y='10-NN accuracy', hue='type', kind='line', data=df)
#plt.savefig("./final_results/lmnn_10_nn.png")
