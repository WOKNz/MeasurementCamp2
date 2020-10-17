import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

intr = pd.read_csv('Baselines geo_internal.csv')
extr = pd.read_csv('Baslines geo_external.csv')

net = pd.concat([intr, extr])
net.to_csv('names_of_lb_gps.csv')
net_duplicate = net.duplicated(subset=["from", "to"])

points_names = net[["from", "to"]].values
net_lb = net[["dX", "dy", "dz", "slope dist"]].values
lb_before_r = (net[["dX", "dy", "dz"]].values)
p = (32 + 46 / 60 + 44.39565 / 3600) * np.pi / 180
l = (35 + 1 / 60 + 23.24213 / 3600) * np.pi / 180
r = np.array([[-np.sin(p) * np.cos(l), -np.sin(p) * np.sin(l), np.cos(p)],
              [-np.sin(l), np.cos(l), 0],
              [np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(l)]])
# print(r)
lb_after_r = np.dot(r, lb_before_r.T).T

lb = np.expand_dims(lb_after_r.flatten(), axis=1)
np.savetxt('lb_excel_GPS.csv', lb_after_r, delimiter=',')

points_names = np.unique(points_names).tolist()
points_names.remove('C1')

dict_index = {}
for i, point in enumerate(points_names):
	dict_index.update({point: i * 3})

A = np.zeros((net.shape[0] * 3, len(points_names) * 3), dtype=int)
P = []
l0 = []
for row in range(0, net.shape[0]):
	frm = net.iloc[row, 0]
	to = net.iloc[row, 1]

	if frm == 'C1':
		# lb[row*3,0] = lb[row*3,0] + 4000
		# lb[row*3+1,0] = lb[row*3+1,0] + 7000
		l0.extend([-4000, -6000, -220])

		A[row * 3, dict_index[to]] = 1
		A[row * 3 + 1, dict_index[to] + 1] = 1
		A[row * 3 + 2, dict_index[to] + 2] = 1


	elif to == 'C1':
		# lb[row*3,0] = lb[row*3,0] - 4000
		# lb[row*3+1,0] = lb[row*3+1,0] - 7000
		l0.extend([4000, 6000, 220])

		A[row * 3, dict_index[frm]] = -1
		A[row * 3 + 1, dict_index[frm] + 1] = -1
		A[row * 3 + 2, dict_index[frm] + 2] = -1

	else:
		l0.extend([0, 0, 0])
		A[row * 3, dict_index[to]] = 1
		A[row * 3 + 1, dict_index[to] + 1] = 1
		A[row * 3 + 2, dict_index[to] + 2] = 1

		A[row * 3, dict_index[frm]] = -1
		A[row * 3 + 1, dict_index[frm] + 1] = -1
		A[row * 3 + 2, dict_index[frm] + 2] = -1
	dn = lb_after_r[row, 0]
	de = lb_after_r[row, 1]
	s_plane = np.sqrt(dn ** 2 + de ** 2)
	du = lb_after_r[row, 2]
	P.append(0.005 ** 2 + (s_plane / 1000000) ** 2)
	P.append(0.005 ** 2 + (s_plane / 1000000) ** 2)
	P.append(0.01 ** 2 + (2 * du / 1000000) ** 2)

l0 = np.expand_dims(np.array(l0), axis=0).T
l = lb - l0
l_lb_l0 = pd.DataFrame(np.hstack((l.reshape((net.shape[0], 3)),
                                  lb.reshape((net.shape[0], 3)),
                                  l0.reshape((net.shape[0], 3)))), columns=['L n(m)', 'L e(m)', 'L u(m)',
                                                                            'Lb n(m)', 'Lb e(m)', 'Lb u(m)',
                                                                            'L0 n(m)', 'L0 e(m)', 'L0 u(m)'])
P = 10 ** (-6) * np.linalg.inv(np.diag(P))
np.savetxt('P_excel_GPS.csv', P, delimiter=',')

# P = np.diag([1,2,3]*2)
N = np.dot(np.dot(A.T, P), A)
u = np.dot(np.dot(A.T, P), l)

x_flat = np.dot(np.linalg.inv(N), u)
x = np.dot(np.linalg.inv(N), u).reshape((len(points_names), 3))
x_df = pd.DataFrame(x, index=points_names, columns=['n(m)', 'e(m)', 'u(m)'])

v = np.dot(A, x_flat) - l
v_df = pd.DataFrame(v, index=None, columns=['V(m)'])

sigma_post = np.dot(np.dot(v.T, P), v) / (net.shape[0] * 3 - (len(points_names) * 3))
error = np.diagonal(sigma_post[0, 0] * np.linalg.inv(N))
error_df = pd.DataFrame(1000 * error.reshape((len(points_names), 3)), index=points_names,
                        columns=['n(mm)', 'e(mm)', 'u(mm)'])

plt.scatter(x_df['e(m)'].values, x_df['n(m)'].values)
plt.show()

# sig_post = np.dot()

if True:
	# np.savetxt('output_gps/001_A.csv',A,delimiter=',')
	# l_lb_l0.to_csv('output_gps/002_l_all.csv')
	# np.savetxt('output_gps/003_P.csv', P, delimiter=',')
	# x_df.to_csv('output_gps/004_xa.csv')
	# error_df.to_csv('output_gps/005_error.csv')
	# v_df.to_csv('output_gps/006_v.csv')
	# np.savetxt('output_gps/007_sigma_post.csv', np.array(sigma_post), delimiter=',')

	pass

print('pause')
