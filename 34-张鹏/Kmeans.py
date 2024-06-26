import numpy as np  
from scipy.spatial.distance import cdist  
import cv2

def kmeans(data, k, max_iters=100, random_state=None):  
    # 1. 初始化聚类中心  
    if random_state is not None:  
        np.random.seed(random_state)  
    # 随机选择k个数据点作为初始聚类中心  
    centroids_idx = np.random.choice(len(data), k, replace=False)  
    centroids = data[centroids_idx]  
  
    for _ in range(max_iters):  
        # 2. 将每个数据点分配给最近的聚类中心  
        labels = np.argmin(cdist(data, centroids, 'euclidean'), axis=1)  
  
        # 3. 重新计算聚类中心  
        new_centroids = np.array([data[labels == i].mean(0) for i in range(k)])  
  
        # 4. 检查聚类中心是否改变  
        if np.all(centroids == new_centroids):  
            break  
  
        centroids = new_centroids  
  
    return centroids, labels  
  