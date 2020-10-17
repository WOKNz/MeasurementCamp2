# %%
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('seaborn-whitegrid')
from functions2 import *

# import lb
lb_df = pd.read_csv('lb_fixed_v2.csv')

from_lb = lb_df['FROM'].tolist()
to_lb = lb_df['TO'].tolist()
lb_np = np.expand_dims(lb_df['VALUE'].to_numpy(), axis=1)

# P calculating
# P = block_diag(np.eye(8) * 0.01, np.eye(16) * 500, np.eye(1) * 0.01)
P = pd.read_csv('P_excel.csv', header=None).to_numpy()

# P = np.eye(25)

# calculating x0
x0 = pd.read_csv('x0_initial.csv', index_col=False)
x0_np = xdf2xnp(x0)
dict_x = {'c2': [0, 1], 'c3': [2, 3], 'c4': [4, 5], 'c5': [6, 7], 'c1_ori': 8, 'c2_ori': 9, 'c3_ori': 10, 'c4_ori': 11,
          'c5_ori': 12}

# calculate l0
l0 = solve_l0(x0_np, from_lb, to_lb, dict_x)
# l0.to_csv('l0_first_test2.csv')

# calculating la
la = np.expand_dims(lb_df.iloc[:, 3].to_numpy(), axis=0).T - l0

# calculatin a
A = a_calc(x0_np, from_lb, to_lb, dict_x)

# calculating N and u
N = np.dot(np.dot(A.T, P), A)
u = np.dot(np.dot(A.T, P), la)

dx = np.dot(np.linalg.inv(N), u)

x0 = x0_np + dx

v = np.dot(A, dx) - la
sig_post = np.dot(np.dot(v.T, P), v) / (25 - 13)
error_full = sig_post * np.linalg.inv(N)
error = np.sqrt(sig_post * np.linalg.inv(N).diagonal())
test_final = np.dot(np.dot(A.T, P), v)

ellipses = getEllipses(error_full[:8, :8], 2)

if True:
	# np.savetxt('output/004_P.csv', P.diagonal(), delimiter=',')
	# np.savetxt('output/005_x0.csv',x0_np,delimiter=',')
	# np.savetxt('output/002_l0.csv',l0,delimiter=',')
	# np.savetxt('output/003_la.csv',la,delimiter=',')
	# np.savetxt('output/001_A.csv',A,delimiter=',')
	# np.savetxt('output/006_dx1.csv',dx,delimiter=',')
	# np.savetxt('output/007_xa1.csv',x0,delimiter=',')
	# np.savetxt('output/009_v1.csv',v,delimiter=',')
	# np.savetxt('output/sig_post.csv',sig_post,delimiter=',')
	# np.savetxt('output/008_error.csv',error.T,delimiter=',')
	# np.savetxt('output/test_final.csv',test_final,delimiter=',')
	# np.savetxt('output/ellipses.csv',ellipses,delimiter=',')
	pass

# loop
# i = 0

if ((np.max(np.abs(dx[0:8, 0])) < 0.0001) and (np.max(np.abs(dx[8:, 0])) < (0.1 / 3600) * np.pi / 180)):
	while not ((np.max(np.abs(dx[0:8, 0])) < 0.00001) and (np.max(np.abs(dx[8:, 0])) < (0.1 / 3600) * np.pi / 180)):
		# if i == 10:
		# 	break
		# i += 1
		l0 = solve_l0(x0, from_lb, to_lb, dict_x)

		A = a_calc(x0, from_lb, to_lb, dict_x)
		la = np.expand_dims(lb_df.iloc[:, 3].to_numpy(), axis=0).T - l0

		N = np.dot(np.dot(A.T, P), A)
		u = np.dot(np.dot(A.T, P), la)

		dx = np.dot(np.linalg.inv(N), u)

		x0 = x0 + dx

		v = np.dot(A, dx) - la

		print(np.dot(v.T, v)[0, 0])

	# print(np.max(np.abs(dx[0:8, 0])))
	# print(np.max(np.abs(dx[8:, 0])))
	# print((np.max(np.abs(dx[0:8, 0])) < 0.00001))
	# print((np.max(np.abs(dx[8:, 0])) < (0.1 / 3600) * np.pi / 180))

else:
	print('One iteration only')

xs = [6000]
xs.extend((x0[0:8:2].flatten()).tolist())
ys = [4000]
ys.extend((x0[1:8:2].flatten()).tolist())

plt.plot(xs, ys, 'o', color='black')
for x, y, label in zip(xs, ys, ['C1', 'C2', 'C3', 'C4', 'C5']):
	plt.annotate(label,  # this is the text
	             (x, y),  # this is the point to label
	             textcoords="offset points",  # how to position the text
	             xytext=(0, 10),  # distance from text to points (x,y)
	             ha='center')  # horizontal alignment can be left, right or center

plt.savefig('draft_plot.png', dpi=300)
plt.show()

print('Done')

# %%
