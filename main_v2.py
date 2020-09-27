import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from functions import *

# defining variables
x_tag = ['c2_e', 'c2_n', 'c3_e', 'c3_n',
         'c4_e', 'c4_n', 'c5_e', 'c5_n',
         'o1', 'o2', 'o3', 'o4', 'o5']

# import lb
lb_df = pd.read_csv('lb_fixed_v2.csv')

from_lb = lb_df['FROM'].tolist()
to_lb = lb_df['TO'].tolist()
lb_np = np.expand_dims(lb_df['VALUE'].to_numpy(), axis=1)

# P calculating
P = block_diag(np.eye(8) * 0.01, np.eye(16) * 500, np.eye(1) * 0.01)

# calculating x0
x0 = solve_x0_v2(lb_df)

# calculate l0
l0 = solve_l0(x0, lb_df)

# calculating la
la = lb_df.iloc[:, 3].to_numpy() - l0.iloc[:, 0].to_numpy()
la = np.expand_dims(la, axis=0).T

# calculatin a
A = a_calc(x0, lb_df)

# calculating N and u
N = np.dot(np.dot(A.T, P), A)
u = np.dot(np.dot(A.T, P), la)

dx = np.dot(np.linalg.inv(N), u)

x0 = updateXdf(dx, x0)
v = np.dot(A, dx) - la

# loop
i = 0
while not ((np.max(np.abs(dx[0:8, 0])) < 0.001) and (np.max(np.abs(dx[8:, 0])) < (5.0 / 3600) * np.pi / 180)):
	i += 1
	l0 = solve_l0(x0, lb_df)

	A = a_calc(x0, lb_df)
	la = lb_df.iloc[:, 3].to_numpy() - l0.iloc[:, 0].to_numpy()
	la = np.expand_dims(la, axis=0).T
	# la_story = np.hstack((la_story,la))

	N = np.dot(np.dot(A.T, P), A)
	u = np.dot(np.dot(A.T, P), la)

	dx = np.dot(np.linalg.inv(N), u)
	# dx_story = np.hstack((dx_story,dx))

	x0 = updateXdf(dx, x0)
	# x0_story = np.hstack((x0_story,xdf2xnp(x0)))
	# fig = px.scatter(x=x0.iloc[0:4, 1].to_list(), y=x0.iloc[0:4, 2].to_list(), text=['c3','c4','c5','c2'])
	# fig.show()

	v = np.dot(A, dx) - la
	# v_story = np.hstack((v_story,v))
	print(np.dot(v.T, v))

print('pause')
