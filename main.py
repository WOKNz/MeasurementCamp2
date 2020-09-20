import sympy as sp
import numpy as np
import pandas as pd
from sympy import init_printing

init_printing()

# na, nb, ea, eb = sp.symbols('na nb ea eb')
# dn_ab = nb-na
# de_ab = eb-ea
# s_ab = sp.sqrt((dn_ab)**2+(de_ab)**2)
# sp.pprint(sp.diff(s_ab,na))
# sp.pprint((sp.diff(s_ab,na)).evalf(subs={na:3, nb:2, eb:0, ea:0}))

lb_df = pd.read_csv('lb_fixed.csv')
rows = lb_df.shape[0]


def azimuth(from_c, to_c, calc_x):
	if from_c != 'c1':
		E_from = calc_x[(calc_x['POINT'] == from_c)].E.iloc[0]
		N_from = calc_x[(calc_x['POINT'] == from_c)].N.iloc[0]
	else:
		E_from = 0
		N_from = 0

	if to_c != 'c1':
		E_to = calc_x[(calc_x['POINT'] == to_c)].E.iloc[0]
		N_to = calc_x[(calc_x['POINT'] == to_c)].N.iloc[0]
	else:
		E_to = 0
		N_to = 0

	dE = E_to - E_from
	dN = N_to - N_from

	azi = np.arctan2(dE, dN)
	# if azi < 0:
	# 	azi = azi + 2 * np.pi
	azi_deg = azi * 180 / np.pi

	return azi


def solve_x0(l):
	c1_values = l[(l['FROM'] == "c1") & (l['TYPE'] == "dir")]
	azi_fix = l[l['TYPE'] == "azi"]['VALUE'].to_numpy()[0]
	azi_fix = c1_values[(c1_values['TO'] == "c3")]['VALUE'].to_numpy()[0] - azi_fix
	c1_values.loc[:, 'VALUE'] = c1_values['VALUE'] - azi_fix
	print(c1_values)

	calc_x = pd.DataFrame(columns=['POINT', 'E', 'N', 'ORI', 'TYPE'])
	rows_x = c1_values.shape[0]
	for row in range(0, rows_x):
		N = np.cos(c1_values.iloc[row, 3]) * l[(l['FROM'] == 'c1') &
		                                       (l['TO'] == c1_values.iloc[row, 1]) &
		                                       (l['TYPE'] == 'dis')]['VALUE'].to_numpy()[0]
		E = np.sin(c1_values.iloc[row, 3]) * l[(l['FROM'] == 'c1') &
		                                       (l['TO'] == c1_values.iloc[row, 1]) &
		                                       (l['TYPE'] == 'dis')]['VALUE'].to_numpy()[0]
		calc_x = calc_x.append({'POINT': c1_values.iloc[row, 1], 'E': E, 'N': N, 'TYPE': 'cor'}, ignore_index=True)

	test = []
	for station in ['c1', 'c2', 'c3', 'c4', 'c5']:

		dirs = l[(l['FROM'] == station) & (l['TYPE'] == 'dir')]
		dir_min = dirs[dirs['VALUE'] == dirs.VALUE.min()]
		from_c = dir_min.FROM.iloc[0]
		to_c = dir_min.TO.iloc[0]

		azi = azimuth(from_c, to_c, calc_x)
		# if azi < 0:
		# 	azi = azi + 2*np.pi

		dir_min = dir_min.VALUE.iloc[0]
		test.append([from_c, to_c, azi * 180 / np.pi, dir_min * 180 / np.pi])

		o_c = None
		if azi < 0:
			o_c = azi + 2 * np.pi - dir_min
		elif (azi > 0) & (azi < dir_min):
			o_c = azi - dir_min

		calc_x = calc_x.append({'POINT': station, 'ORI': o_c, 'TYPE': 'ori'}, ignore_index=True)
	print(pd.DataFrame(test))
	print(calc_x)
	return calc_x


x0 = solve_x0(lb_df)


def solve_l0(x0, lb):
	l0 = pd.DataFrame(columns=['VALUE'])
	rows_l0 = lb.shape[0]

	for row in range(0, rows_l0):

		if lb.iloc[row, 0] == 'c1':
			from_e = 0
			from_n = 0
		else:
			from_e = x0[(x0['POINT'] == lb.iloc[row, 0])]['E'].to_numpy()[0]
			from_n = x0[(x0['POINT'] == lb.iloc[row, 0])]['N'].to_numpy()[0]

		if lb.iloc[row, 1] == 'c1':
			to_e = 0
			to_n = 0
		else:
			to_e = x0[(x0['POINT'] == lb.iloc[row, 1])]['E'].to_numpy()[0]
			to_n = x0[(x0['POINT'] == lb.iloc[row, 1])]['N'].to_numpy()[0]

		if lb.iloc[row, 2] == 'dis':

			l0 = l0.append({'VALUE': np.sqrt((to_e - from_e) ** 2 + (to_n - from_n) ** 2)}, ignore_index=True)

		elif lb.iloc[row, 2] == 'dir':

			ori_station_name = lb.iloc[row, 0]
			ori_station = x0[(x0['POINT'] == ori_station_name) &
			                 (x0['TYPE'] == 'ori')].ORI.iloc[0]

			from_c = lb.iloc[row, 0]
			to_c = lb.iloc[row, 1]

			azi = azimuth(from_c, to_c, x0)
			b = 0
			if azi - ori_station < 0:
				b = azi - ori_station + 2 * np.pi
			else:
				b = azi - ori_station

			l0 = l0.append({'VALUE': b}, ignore_index=True)

		else:
			l0 = l0.append({'VALUE': np.arctan2(to_e, to_n)}, ignore_index=True)  # azimuth option

	# l0 = l0.append({'VALUE': val}, ignore_index=True)

	print(l0)

	return l0


l0 = solve_l0(x0, lb_df)


def xdf2xnp(x_df):
	cors = x_df[x_df['TYPE'] == 'cor']
	cors = cors.iloc[:, 1:3].to_numpy()
	cors = cors.flatten()

	oris = x_df[x_df['TYPE'] == 'ori']
	oris = oris.iloc[:, 3].to_numpy()
	oris = oris.flatten()

	return np.expand_dims(np.concatenate((cors, oris), axis=0), axis=0).T


x0_np = xdf2xnp(x0)


# Compute A
def a_calc(calc_x, lb_df):
	A = np.zeros((lb_df.shape[0], 13))
	dict = {'c3': 0, 'c4': 2, 'c5': 4, 'c2': 6}
	dict_ori = {'c1': 8, 'c2': 9, 'c3': 10, 'c4': 11, 'c5': 12}

	for row in range(0, lb_df.shape[0]):

		from_c = lb_df.iloc[row, 0]
		to_c = lb_df.iloc[row, 1]

		if from_c != 'c1':
			E_from = calc_x[(calc_x['POINT'] == from_c)].E.iloc[0]
			N_from = calc_x[(calc_x['POINT'] == from_c)].N.iloc[0]
		else:
			E_from = 0
			N_from = 0

		if to_c != 'c1':
			E_to = calc_x[(calc_x['POINT'] == to_c)].E.iloc[0]
			N_to = calc_x[(calc_x['POINT'] == to_c)].N.iloc[0]
		else:
			E_to = 0
			N_to = 0

		dE = E_to - E_from
		dN = N_to - N_from

		s2 = dE ** 2 + dN ** 2
		s = np.sqrt(dE ** 2 + dN ** 2)

		if lb_df.iloc[row, 2] == 'dis':
			part_e_a = -dE / s
			part_n_a = -dN / s
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_c != 'c1':
				A[row, dict[from_c]] = part_e_a
				A[row, dict[from_c] + 1] = part_n_a
			if to_c != 'c1':
				A[row, dict[to_c]] = part_e_b
				A[row, dict[to_c] + 1] = part_n_b

		if lb_df.iloc[row, 2] == 'dir':
			part_e_a = -dE / s2
			part_n_a = dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_c != 'c1':
				A[row, dict[from_c]] = part_e_a
				A[row, dict[from_c] + 1] = part_n_a
			if to_c != 'c1':
				A[row, dict[to_c]] = part_e_b
				A[row, dict[to_c] + 1] = part_n_b

			A[row, dict_ori[from_c]] = -1

		if lb_df.iloc[row, 2] == 'azi':
			part_e_a = -dE / s2
			part_n_a = dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_c != 'c1':
				A[row, dict[from_c]] = part_e_a
				A[row, dict[from_c] + 1] = part_n_a
			if to_c != 'c1':
				A[row, dict[to_c]] = part_e_b
				A[row, dict[to_c] + 1] = part_n_b

	return A


A = a_calc(x0, lb_df)

lb_df.to_csv('original_lb.csv')
l0.to_csv('original_l0_from_x0.csv')

la = lb_df.iloc[:, 3].to_numpy() - l0.iloc[:, 0].to_numpy()
la = np.expand_dims(la, axis=0).T

N = np.dot(A.T, A)
u = np.dot(A.T, la)

dx = np.dot(np.linalg.inv(N), u)

print('wait')
