import numpy as np
import pandas as pd
import scipy.linalg
from functions3 import *

# Lodaing GPS vectors and organazing them

intr = pd.read_csv('data_gps_norm/Baselines geo_internal.csv')
extr = pd.read_csv('data_gps_norm/Baslines geo_external.csv')

net = pd.concat([intr, extr])
# is_duplicates = net.duplicated(subset=["from","to"])

# points_names = net[["from","to"]].values
net_lb = net[["dX", "dy", "dz"]].values
lb_before_r = (net[["dX", "dy", "dz"]].values)
p = (32 + 46 / 60 + 44.39565 / 3600) * np.pi / 180
l = (35 + 1 / 60 + 23.24213 / 3600) * np.pi / 180
r = np.array([[-np.sin(p) * np.cos(l), -np.sin(p) * np.sin(l), np.cos(p)],
              [-np.sin(l), np.cos(l), 0],
              [np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(l)]])

lb_after_r = np.dot(r, lb_before_r.T).T

final_gps_lb = net[["from", "to", "dX", "dy", "dz"]].copy()
final_gps_lb.loc[:, ["dX", "dy", "dz"]] = lb_after_r.copy()
final_gps_lb.columns = ['from', 'to', 'dn', 'de', 'du']
# final_gps_lb['from'] = final_gps_lb['from'].str.lower()

del intr, extr, net, net_lb, lb_before_r, lb_after_r, l, p, r

# Classical mesurments and organazing them
lb_df = pd.read_csv('data_gps_norm/lb_fixed_v2_gps.csv')
lb_df['FROM'] = lb_df['FROM'].str.upper()
lb_df['TO'] = lb_df['TO'].str.upper()

for index, row in final_gps_lb.iterrows():
	lb_df = lb_df.append({'FROM': row.loc['from'],
	                      'TO': row.loc['to'],
	                      'TYPE': 'dn',
	                      'VALUE': row.loc['dn'],
	                      'dim': 'meter'}, ignore_index=True)
	lb_df = lb_df.append({'FROM': row.loc['from'],
	                      'TO': row.loc['to'],
	                      'TYPE': 'de',
	                      'VALUE': row.loc['de'],
	                      'dim': 'meter'}, ignore_index=True)
	lb_df = lb_df.append({'FROM': row.loc['from'],
	                      'TO': row.loc['to'],
	                      'TYPE': 'du',
	                      'VALUE': row.loc['du'],
	                      'dim': 'meter'}, ignore_index=True)

del row, index

# Extracting names for points
points_names = lb_df[["FROM", "TO"]].values
points_names = np.unique(points_names).tolist()
points_names.remove('C1')

del final_gps_lb

# Creating P
P_clasic = pd.read_csv('data_gps_norm/P_classic.csv', header=None).to_numpy()
P_gps = pd.read_csv('data_gps_norm/P_gps.csv', header=None).to_numpy()
P = scipy.linalg.block_diag(P_clasic, P_gps)

del P_gps, P_clasic

# Import x0

x0 = pd.read_csv('data_gps_norm/x0_initial.csv')
x0['POINT'] = x0['POINT'].str.upper()

# Examples
# c2_east = x0.loc[x0['POINT'] == 'C2', 'E'].values[0]
# c2_ori = x0.loc[(x0['POINT'] == 'C2') & (x0['TYPE'] == 'ori'), 'ORI'].values[0]

max_dx = 1
threshold = (0.1 / 3600) * np.pi / 180

while max_dx > threshold:
	L0_df = calculateL0(x0, lb_df)

	L_df, L_np = calculateL(lb_df, L0_df)

	A_df, A_np = calculateA(x0, L_df, points_names)

	N = np.dot(np.dot(A_np.T, P), A_np)
	N_inv = np.linalg.inv(N)

	u = np.dot(np.dot(A_np.T, P), L_np)

	dx = np.dot(N_inv, u)

	max_dx = np.max(np.abs(dx))

	x0 = updateX(x0, dx, A_df)

	print('Iteration done, max dx val = ', max_dx)

v = np.dot(A_np, dx) - L_np
v_df = lb_df.copy()
v_df['VALUE'] = pd.DataFrame(v)
sig_post2 = np.dot(np.dot(v.T, P), v) / (L_np.shape[0] - dx.shape[0])
sig_post = np.sqrt(np.dot(np.dot(v.T, P), v) / (L_np.shape[0] - dx.shape[0]))
error = np.diagonal(sig_post2[0, 0] * N_inv)
error = np.sqrt(error)
error = np.insert(error, 38, None)
points_error = np.expand_dims(error[0:-5], axis=1)
points_error = np.round(points_error * 100, 1)
ori_error = np.expand_dims(error[51:], axis=1)
points_error = pd.DataFrame(points_error.reshape((len(points_names)), 3), index=points_names,
                            columns=['N(cm)', 'E(cm)', 'U(cm)'])
ori_error = pd.DataFrame(np.round(ori_error * 180 * 3600 / np.pi, 2), index=['C1', 'C2', 'C3', 'C4', 'C5'],
                         columns=['Deg (sec)'])

if True:
# A_df.to_csv('output_gps_norm/001_A.csv')
# lb_df.to_csv('output_gps_norm/002_lb.csv')
# L0_df.to_csv('output_gps_norm/002_l0.csv')
# np.savetxt('output_gps_norm/003_P.csv', P, delimiter=',')
# x0.to_csv('output_gps_norm/004_xa.csv')
# points_error.to_csv('output_gps_norm/005_error_points.csv')
# ori_error.to_csv('output_gps_norm/005_error_ori.csv')
# v_df.to_csv('output_gps_norm/006_v.csv')
# np.savetxt('output_gps_norm/007_sigma_post.csv', sig_post, delimiter=',')
# pass
print('debug')
