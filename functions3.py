import numpy as np
import pandas as pd


def calculateL0(x_df, Lb_df):
	L0_df = []
	for index, row in Lb_df.iterrows():
		FROM = row['FROM']
		TO = row['TO']
		TYPE = row['TYPE']
		VALUE = row['VALUE']

		if TYPE == 'dis':
			if FROM == 'C1':
				from_e = 6000
				from_n = 4000
			else:
				from_e = x_df.loc[x_df['POINT'] == FROM, 'E'].values[0]
				from_n = x_df.loc[x_df['POINT'] == FROM, 'N'].values[0]

			if TO == 'C1':
				to_e = 6000
				to_n = 4000
			else:
				to_e = x_df.loc[x_df['POINT'] == TO, 'E'].values[0]
				to_n = x_df.loc[x_df['POINT'] == TO, 'N'].values[0]

			dE = to_e - from_e
			dN = to_n - from_n

			s = np.sqrt(dE ** 2 + dN ** 2)
			L0_df.append(s)

		if TYPE == 'dir':
			azi = azimuth(x_df, row)
			x_ori = x_df.loc[(x_df['POINT'] == row['FROM']) & (x_df['TYPE'] == 'ori'), 'ORI'].values[0]
			direction = azi - x_ori

			if direction < 0:
				direction = direction + 2 * np.pi

			L0_df.append(direction)

		if TYPE == 'azi':
			azi = azimuth(x_df, row)
			L0_df.append(azi)

		if TYPE == 'dn':

			if FROM == 'C1':
				from_n = 4000
			else:
				from_n = x_df.loc[(x_df['POINT'] == FROM) & (x_df['TYPE'] == 'cor'), 'N'].values[0]

			if TO == 'C1':
				to_n = 4000
			else:
				to_n = x_df.loc[(x_df['POINT'] == TO) & (x_df['TYPE'] == 'cor'), 'N'].values[0]

			L0_df.append(to_n - from_n)

		if TYPE == 'de':

			if FROM == 'C1':
				from_e = 6000
			else:
				from_e = x_df.loc[(x_df['POINT'] == FROM) & (x_df['TYPE'] == 'cor'), 'E'].values[0]

			if TO == 'C1':
				to_e = 6000
			else:
				to_e = x_df.loc[(x_df['POINT'] == TO) & (x_df['TYPE'] == 'cor'), 'E'].values[0]

			L0_df.append(to_e - from_e)
		if TYPE == 'du':
			if (FROM == 'C4') or (TO == 'C4'):
				continue

			if FROM == 'C1':
				from_u = 220
			else:
				from_u = x_df.loc[(x_df['POINT'] == FROM) & (x_df['TYPE'] == 'cor'), 'U'].values[0]

			if TO == 'C1':
				to_u = 220
			else:
				to_u = x_df.loc[(x_df['POINT'] == TO) & (x_df['TYPE'] == 'cor'), 'U'].values[0]

			L0_df.append(to_u - from_u)

	temp = Lb_df.copy()
	temp['VALUE'] = np.array(L0_df)
	L0_df = temp

	return L0_df


def calculateL(Lb_df, L0_df):
	temp_df = Lb_df.copy()
	temp_df['VALUE'] = temp_df['VALUE'] - L0_df['VALUE']
	temp_np = temp_df['VALUE'].to_numpy()
	temp_np = np.expand_dims(temp_np, axis=1)

	return temp_df, temp_np


def calculateA(x_df, L_df, x_names):
	header = pd.MultiIndex.from_product([x_names, ['N', 'E', 'U']], names=['POINT', 'TYPE'])
	ori_points = ['C1', 'C2', 'C3', 'C4', 'C5']
	header2 = pd.MultiIndex.from_product([ori_points, ['ORI']], names=['POINT', 'TYPE'])
	A = pd.DataFrame(np.zeros((L_df.shape[0], len(x_names) * 3)), columns=header)
	A2 = pd.DataFrame(np.zeros((L_df.shape[0], len(ori_points))), columns=header2)
	A.drop(('C4', 'U'), axis=1, inplace=True)
	# A = pd.concat([A,A2],ignore_index=True)
	A = pd.merge(A, A2, left_index=True, right_index=True)
	# A.to_csv('output_gps_norm/test.csv')

	for index, row in L_df.iterrows():
		# if (row['FROM'] != 'C1') and (row['TO'] != 'C1'):
		# 	A.loc[index,(row['FROM'],'N')] = row['VALUE']

		FROM = row['FROM']
		TO = row['TO']
		TYPE = row['TYPE']
		VALUE = row['VALUE']

		if FROM == 'C1':
			from_e = 6000
			from_n = 4000
		else:
			from_e = x_df.loc[x_df['POINT'] == FROM, 'E'].values[0]
			from_n = x_df.loc[x_df['POINT'] == FROM, 'N'].values[0]

		if TO == 'C1':
			to_e = 6000
			to_n = 4000
		else:
			to_e = x_df.loc[x_df['POINT'] == TO, 'E'].values[0]
			to_n = x_df.loc[x_df['POINT'] == TO, 'N'].values[0]

		dE = to_e - from_e
		dN = to_n - from_n

		s2 = dE ** 2 + dN ** 2
		s = np.sqrt(dE ** 2 + dN ** 2)

		if TYPE == 'dis':

			part_e_a = -dE / s
			part_n_a = -dN / s
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if FROM != 'C1':
				A.loc[index, (row['FROM'], 'N')] = part_n_a
				A.loc[index, (row['FROM'], 'E')] = part_e_a
			if TO != 'C1':
				A.loc[index, (row['TO'], 'N')] = part_n_b
				A.loc[index, (row['TO'], 'E')] = part_e_b

		if TYPE == 'dir':

			part_n_a = dE / s2
			part_e_a = -dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if FROM != 'C1':
				A.loc[index, (row['FROM'], 'N')] = part_n_a
				A.loc[index, (row['FROM'], 'E')] = part_e_a
			if TO != 'C1':
				A.loc[index, (row['TO'], 'N')] = part_n_b
				A.loc[index, (row['TO'], 'E')] = part_e_b

			A.loc[index, (row['FROM'], 'ORI')] = -1

		if TYPE == 'azi':

			part_n_a = dE / s2
			part_e_a = -dN / s2
			part_e_b = -part_e_a
			part_n_b = -part_n_a

			if FROM != 'C1':
				A.loc[index, (row['FROM'], 'N')] = part_n_a
				A.loc[index, (row['FROM'], 'E')] = part_e_a
			if TO != 'C1':
				A.loc[index, (row['TO'], 'N')] = part_n_b
				A.loc[index, (row['TO'], 'E')] = part_e_b

		if TYPE == 'dn':
			if FROM != 'C1':
				A.loc[index, (row['FROM'], 'N')] = -1
			if TO != 'C1':
				A.loc[index, (row['TO'], 'N')] = 1

		if TYPE == 'de':
			if FROM != 'C1':
				A.loc[index, (row['FROM'], 'E')] = -1
			if TO != 'C1':
				A.loc[index, (row['TO'], 'E')] = 1

		if TYPE == 'du':
			if (FROM != 'C1') and (FROM != 'C4'):
				A.loc[index, (row['FROM'], 'U')] = -1
			if (TO != 'C1') and (TO != 'C4'):
				A.loc[index, (row['TO'], 'U')] = 1

	return A, A.to_numpy()


# return A_df, A_np

def azimuth(x_df, row):
	FROM = row['FROM']
	TO = row['TO']
	TYPE = row['TYPE']
	VALUE = row['VALUE']

	if FROM == 'C1':
		from_e = 6000
		from_n = 4000
	else:
		from_e = x_df.loc[x_df['POINT'] == FROM, 'E'].values[0]
		from_n = x_df.loc[x_df['POINT'] == FROM, 'N'].values[0]

	if TO == 'C1':
		to_e = 6000
		to_n = 4000
	else:
		to_e = x_df.loc[x_df['POINT'] == TO, 'E'].values[0]
		to_n = x_df.loc[x_df['POINT'] == TO, 'N'].values[0]

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


def xNp2xDf(x_np, x_df, A_df):
	x_df_copy = x_df.copy()
	for index in range(0, x_np.shape[0]):
		dx_value = x_np[index, 0]
		row_df = A_df.columns[index][0]
		column_df = A_df.columns[index][1]
		row_type = None
		if column_df == 'N' or column_df == 'E' or column_df == 'U':
			row_type = 'cor'
		else:
			row_type = 'ori'

		x_df_copy.loc[(x_df_copy['POINT'] == row_df) & (x_df_copy['TYPE'] == row_type), column_df] = dx_value

	return x_df_copy


def updateX(x0, dx, A_df):
	x_df_copy = xNp2xDf(dx, x0, A_df)
	x0_copy = x0.copy()
	x0_copy[['N', 'E', 'U', 'ORI']] = x0_copy[['N', 'E', 'U', 'ORI']] + x_df_copy[['N', 'E', 'U', 'ORI']]
	return x0_copy
