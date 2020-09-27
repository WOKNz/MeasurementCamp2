import numpy as np
import pandas as pd
import plotly.express as px
from scipy.linalg import block_diag
from functions import *

lb_df = pd.read_csv('lb_fixed_v2.csv')
lb_np = np.expand_dims(lb_df.iloc[:, 3].to_numpy(), axis=1)

x0 = solve_x0(lb_df)
l0 = solve_l0(x0, lb_df)
x0_np = xdf2xnp(x0)
x0_story = x0_np

A = a_calc(x0, lb_df)
A_story = A

P = block_diag(np.eye(8) * 0.01, np.eye(16) * 500, np.eye(1) * 0.01)

# np.savetxt('A.csv',A)

lb_df.to_csv('original_lb.csv')
l0.to_csv('original_l0_from_x0.csv')

la = lb_df.iloc[:, 3].to_numpy() - l0.iloc[:, 0].to_numpy()
la = np.expand_dims(la, axis=0).T
la_story = la

N = np.dot(np.dot(A.T, P), A)
u = np.dot(np.dot(A.T, P), la)

dx = np.dot(np.linalg.inv(N), u)
dx_story = dx

test = np.max(np.abs(dx[0:8, 0]))
test2 = np.max(np.abs(dx[8:, 0]))
x0 = updateXdf(dx, x0)
v = np.dot(A, dx) - la
v_story = v

# fig = px.scatter(x=x0.iloc[0:4,1].to_list(), y=x0.iloc[0:4,2].to_list(), text=['c3','c4','c5','c2'])
# fig.show()

i = 0
while not ((np.max(np.abs(dx[0:8, 0])) < 0.001) and (np.max(np.abs(dx[8:, 0])) < (5.0 / 3600) * np.pi / 180)):
	i += 1
	l0 = solve_l0(x0, lb_df)

	A = a_calc(x0, lb_df)
	A_story = np.dstack((A_story, A))
	la = lb_df.iloc[:, 3].to_numpy() - l0.iloc[:, 0].to_numpy()
	la = np.expand_dims(la, axis=0).T
	la_story = np.hstack((la_story, la))

	N = np.dot(np.dot(A.T, P), A)
	u = np.dot(np.dot(A.T, P), la)

	dx = np.dot(np.linalg.inv(N), u)
	dx_story = np.hstack((dx_story, dx))

	x0 = updateXdf(dx, x0)
	x0_story = np.hstack((x0_story, xdf2xnp(x0)))
	# fig = px.scatter(x=x0.iloc[0:4, 1].to_list(), y=x0.iloc[0:4, 2].to_list(), text=['c3','c4','c5','c2'])
	# fig.show()

	v = np.dot(A, dx) - la
	v_story = np.hstack((v_story, v))
	print(np.max(np.abs(v)))

# #iter2
# x0 = updateXdf(dx,x0)
# l0 = solve_l0(x0, lb_df)
# A = a_calc(x0,lb_df)
# la = lb_df.iloc[:,3].to_numpy()-l0.iloc[:,0].to_numpy()
# la = np.expand_dims(la,axis=0).T
#
# N = np.dot(A.T,A)
# u = np.dot(A.T,la)
#
# dx = np.dot(np.linalg.inv(N),u)
#
# #iter3
# x0 = updateXdf(dx,x0)
# l0 = solve_l0(x0, lb_df)
# A = a_calc(x0,lb_df)
# la = lb_df.iloc[:,3].to_numpy()-l0.iloc[:,0].to_numpy()
# la = np.expand_dims(la,axis=0).T
#
# N = np.dot(A.T,A)
# u = np.dot(A.T,la)
#
# dx = np.dot(np.linalg.inv(N),u)


print('wait')
