import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk_cluster


# clustering    k-means++
object_sizes = pd.read_csv("data/object_sizes.csv")

X = object_sizes[["width", "height"]]

# plt.scatter(x=object_sizes["width"], y=object_sizes["height"])


k_means_pp_model = sk_cluster.KMeans(n_clusters=4)
k_means_pp_model.fit(X)

object_classes = k_means_pp_model.predict(X)

k_means_pp_centroid = k_means_pp_model.cluster_centers_

plt.scatter(x=object_sizes["width"], y=object_sizes["height"], c=object_classes, cmap="gist_rainbow")
plt.scatter(x=k_means_pp_centroid[:, 0], y=k_means_pp_centroid[:, 1], marker="X", color="k", s=100)   # рисование центроиды

plt.show()