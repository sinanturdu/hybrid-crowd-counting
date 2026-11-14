import scipy.io as sio

mat = sio.loadmat("Test/img_0001_ann.mat")
points = mat["annPoints"]

print(f"👥 Ground Truth Kişi Sayısı: {points.shape[0]}")
