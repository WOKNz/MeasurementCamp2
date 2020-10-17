import numpy as np
import pandas as pd


def xdf2xnp(x_df):
	cors = x_df[x_df['TYPE'] == 'cor']
	cors = cors.iloc[:, 1:3].to_numpy()
	cors = cors.flatten()

	oris = x_df[x_df['TYPE'] == 'ori']
	oris = oris.iloc[:, 3].to_numpy()
	oris = oris.flatten()

	return np.expand_dims(np.concatenate((cors, oris), axis=0), axis=0).T


def azimuth(from_c, to_c, dict_x, x0):
	if from_c == 'c1':
		from_e = 6000
		from_n = 4000
	else:
		from_e = x0[dict_x[from_c][0], 0]
		from_n = x0[dict_x[from_c][1], 0]

	if to_c == 'c1':
		to_e = 6000
		to_n = 4000
	else:
		to_e = x0[dict_x[to_c][0], 0]
		to_n = x0[dict_x[to_c][1], 0]

	dE = to_e - from_e
	dN = to_n - from_n

	if (dE > 0) and (dN > 0):
		azi = np.arctan(np.abs(dE / dN))
	elif (dE > 0) and (dN < 0):
		azi = np.pi - np.arctan(np.abs(dE / dN))
	elif (dE < 0) and (dN < 0):
		azi = np.pi + np.arctan(np.abs(dE / dN))
	else:
		azi = 2 * np.pi - np.arctan(np.abs(dE / dN))

	# if azi < 0:
	# 	azi = azi + 2 * np.pi
	azi_deg = azi * 180 / np.pi

	return azi


def solve_l0(x0, from_lb, to_lb, dict_x):
	l_new = []

	for i, row in enumerate(from_lb):
		if i < 8:
			if row == 'c1':
				from_e = 6000
				from_n = 4000
			else:
				from_e = x0[dict_x[from_lb[i]][0], 0]
				from_n = x0[dict_x[from_lb[i]][1], 0]

			if to_lb[i] == 'c1':
				to_e = 6000
				to_n = 4000
			else:
				to_e = x0[dict_x[to_lb[i]][0], 0]
				to_n = x0[dict_x[to_lb[i]][1], 0]

			dE = to_e - from_e
			dN = to_n - from_n

			s = np.sqrt(dE ** 2 + dN ** 2)
			l_new.append(s)



		elif (i > 7) and (i < len(from_lb) - 1):

			azi = azimuth(row, to_lb[i], dict_x, x0)
			direction = azi - x0[dict_x[row + '_ori'], 0]

			if direction < 0:
				direction = direction + 2 * np.pi

			l_new.append(direction)
		else:
			azi = azimuth(row, to_lb[i], dict_x, x0)
			l_new.append(azi)

	return np.expand_dims(np.array(l_new), axis=1)


# Compute A
def a_calc(x0, from_lb, to_lb, dict_x):
	A = np.zeros((len(from_lb), 13))
	dict = {'c2': 0, 'c3': 2, 'c4': 4, 'c5': 6}
	dict_ori = {'c1': 8, 'c2': 9, 'c3': 10, 'c4': 11, 'c5': 12}

	for i in range(0, len(from_lb)):

		if from_lb[i] == 'c1':
			from_e = 6000
			from_n = 4000
		else:
			from_e = x0[dict_x[from_lb[i]][0], 0]
			from_n = x0[dict_x[from_lb[i]][1], 0]

		if to_lb[i] == 'c1':
			to_e = 6000
			to_n = 4000
		else:
			to_e = x0[dict_x[to_lb[i]][0], 0]
			to_n = x0[dict_x[to_lb[i]][1], 0]

		dE = to_e - from_e
		dN = to_n - from_n

		s2 = dE ** 2 + dN ** 2
		s = np.sqrt(dE ** 2 + dN ** 2)

		if i < 8:

			part_e_a = -dE / s
			part_n_a = -dN / s
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_lb[i] != 'c1':
				A[i, dict[from_lb[i]]] = part_e_a
				A[i, dict[from_lb[i]] + 1] = part_n_a
			if to_lb[i] != 'c1':
				A[i, dict[to_lb[i]]] = part_e_b
				A[i, dict[to_lb[i]] + 1] = part_n_b




		elif (i > 7) and (i < len(from_lb) - 1):

			part_n_a = dE / s2
			part_e_a = -dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_lb[i] != 'c1':
				A[i, dict[from_lb[i]]] = part_e_a
				A[i, dict[from_lb[i]] + 1] = part_n_a
			if to_lb[i] != 'c1':
				A[i, dict[to_lb[i]]] = part_e_b
				A[i, dict[to_lb[i]] + 1] = part_n_b

			A[i, dict_ori[from_lb[i]]] = -1

		else:
			part_n_a = dE / s2
			part_e_a = -dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if from_lb[i] != 'c1':
				A[i, dict[from_lb[i]]] = part_e_a
				A[i, dict[from_lb[i]] + 1] = part_n_a
			if to_lb[i] != 'c1':
				A[i, dict[to_lb[i]]] = part_e_b
				A[i, dict[to_lb[i]] + 1] = part_n_b

	return A


def getEllipses(sigma_mat, block_size):
	sg = sigma_mat
	bs = block_size
	list_of_blocks = []
	list_of_eig_angle = []
	direction = []
	final_table = np.zeros((4, 4))
	for i in range(0, sg.shape[0], 2):
		list_of_blocks.append(sg[i:i + 2, i:i + 2])
		axi, vec = np.linalg.eig(list_of_blocks[-1])
		list_of_eig_angle.append([axi, vec])
		dir1 = np.arctan2(list_of_eig_angle[-1][1][0, 0], list_of_eig_angle[-1][1][1, 0])
		dir2 = np.arctan2(list_of_eig_angle[-1][1][0, 1], list_of_eig_angle[-1][1][1, 1])
		direction.append([dir1 * 180 / np.pi, dir2 * 180 / np.pi])
		final_table[int(i / 2), 0] = list_of_eig_angle[-1][0][0]
		final_table[int(i / 2), 1] = list_of_eig_angle[-1][0][1]
		final_table[int(i / 2), 2] = dir1 * 180 / np.pi
		final_table[int(i / 2), 3] = dir2 * 180 / np.pi
	return final_table
