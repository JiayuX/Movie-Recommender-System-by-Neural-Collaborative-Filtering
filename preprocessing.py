import math
import numpy as np
import pandas as pd
import torch


class ImRecomDataPreprocessor():
	"""
			This class is used to preprocess the dataset for 
		building a recommender system with implicit feedback. 
	"""

	def __init__(self):
		self.user_col = str()
		self.item_col = str()
		self.interaction_col = str()
		self.time_col = None
		self.num_users = int()
		self.num_items = int()
		self.timestamps = None
		self.dataset = None
		self.user2idx = dict()
		self.item2idx = dict()
		self.user2posnum_mapping = dict()

	def fit(self, df_orig, user_col, item_col, interaction_col, time_col = None, num_users = None, num_items = None):
		"""
				Fit the preprocessor to a dataset.
				'time_col' is 'None' by default, in which case a random
			positive sample is selected to leave out. If a column name is
			inputed as 'time_col' (which is not 'None', of course), the 
			sample with the lastest interaction is selected to leave out.
		"""

		##### Get the dataset with specified numbers of users and items #####
		self.user_col = user_col
		self.item_col = item_col
		self.interaction_col = interaction_col

		if num_users is None:
			num_users = df_orig[user_col].nunique()

		if num_items is None:
			num_items = df_orig[item_col].nunique()

		self.num_users = num_users
		self.num_items = num_items

		# Get the 'num_users' users with the most interactions
		user_groups = df_orig.groupby(self.user_col)[self.interaction_col].count()
		included_users = user_groups.sort_values(ascending = False)[:num_users]
		
		# Get the 'num_items' items with the most interactions
		item_groups = df_orig.groupby(self.item_col)[self.interaction_col].count()
		included_items = item_groups.sort_values(ascending = False)[:num_items]

		self.time_col = time_col
		if self.time_col:
			columns = [self.user_col, self.item_col, self.interaction_col, self.time_col]
		else:
			columns = [user_col, item_col, interaction_col]

		df = ((df_orig.
			    	join(included_users, rsuffix = '_r', how = 'inner', on = self.user_col).
			    	join(included_items, rsuffix = '_r', how = 'inner', on = self.item_col)).
	    			reset_index()[columns])

		##### Reindex the users and items #####
		unique_users = df[self.user_col].unique()
		self.user2idx = {old: new for new, old in enumerate(unique_users)}
		df[self.user_col] = df[self.user_col].map(self.user2idx)

		unique_items = df[self.item_col].unique()
		self.item2idx = {old: new for new, old in enumerate(unique_items)}
		df[self.item_col] = df[self.item_col].map(self.item2idx)

		##### Get the dataset #####
		inter_table = pd.crosstab(df[self.user_col], df[self.item_col], df[self.interaction_col], aggfunc = np.sum)
		self.dataset = inter_table.reset_index().melt(id_vars = [self.user_col], value_vars = None, var_name = self.item_col, value_name = self.interaction_col, ignore_index = True)
		if self.time_col:
			self.dataset = pd.merge(self.dataset, df[[self.user_col, self.item_col, self.time_col]], how = 'left', on = [self.user_col, self.item_col])
		self.dataset[interaction_col] = self.dataset[interaction_col].apply(lambda x: 0 if math.isnan(x) else 1)

		return self.dataset, (self.num_users, self.num_items), (self.user2idx, self.item2idx)

	def create_dataset(self, neg_per_pos = 4, neg_per_pos_test = 100):
		"""
				Create the dataset for training recommender models.
		"""

		pos_data = self.dataset[self.dataset[self.interaction_col] == 1]
		neg_data = self.dataset[self.dataset[self.interaction_col] == 0]

		print(f'We have {len(neg_data)} negative training data.\n')
		print(f'Before leaving one out, we have {len(pos_data)} positive training data.\n')

		# A dict mapping each user_id to the number of its interacted items
		self.user2posnum_mapping = pos_data.groupby(self.user_col)[self.interaction_col].count().to_dict()
		print(f'The least and most interaction of a user among all users: {min(self.user2posnum_mapping.values())}, {max(self.user2posnum_mapping.values())} \n')

		##### Leave one out #####
		if self.time_col:
			train_data, test_data = self.leave_lastest_one_out(pos_data)
		else:
			train_data, test_data = self.leave_random_one_out(pos_data)

		print('~' * 100)
		print(f'After leaving one out, we have {len(train_data)} positive training data and {len(test_data)} test data.\n')
		print('~' * 100)

		##### Negative sampling #####
		neg_samples = self.negative_sampling(neg_data, neg_per_pos)
		neg_samples_test = self.negative_sampling_test(neg_data, neg_per_pos_test)

		print(f'Before negative sampling, we have {len(train_data)} positive training data and {len(test_data)} positive test data.\n')
		print(f'We should sample {neg_per_pos * sum(self.user2posnum_mapping.values())} negative training data and {neg_per_pos_test * len(test_data)} negative test data.\n')
		print(f'We actually sampled {len(neg_samples)} negitive training data and {len(neg_samples_test)} negative test data.\n')

		train_data = pd.concat([train_data, neg_samples], axis = 0, ignore_index=True)
		test_data = pd.concat([test_data, neg_samples_test], axis = 0, ignore_index=True)
		
		print(f'After negative sampling, we have {len(train_data)} training data and {len(test_data)} test data.\n')

		return (train_data[[self.user_col, self.item_col]], train_data[self.interaction_col]), (test_data[[self.user_col, self.item_col]], test_data[self.interaction_col])

	def leave_lastest_one_out(self, df):
		"""
				Leave the lastest interaction out for each
			user to be the test set.
		"""

		df['group_rank'] = df.groupby([self.user_col])[self.time_col].rank(method = 'first', ascending = False)
		test_data = df[df['group_rank'] == 1]
		train_data = df[df['group_rank'] > 1]

		assert train_data[self.user_col].nunique() == test_data[self.user_col].nunique(), "Warning: the numbers of users in train and test sets don't match!"
		
		return train_data[[self.user_col, self.item_col, self.interaction_col]], test_data[[self.user_col, self.item_col, self.interaction_col]]

	def leave_random_one_out(self, df):
		"""
				Leave a random interaction out for each user 
			to be the test set.
		"""

		# Define a 'tot_id' column as the primary key to facilitate the splitting later
		df.reset_index(drop = True)
		df['tot_id'] = df.index.tolist()

		test_data = pd.DataFrame(df.groupby(self.user_col)[['tot_id', self.item_col, self.interaction_col]].apply(lambda x: x.sample(1))).reset_index(drop = False)[['tot_id', self.user_col, self.item_col, self.interaction_col]]

		train_data = df[~df['tot_id'].isin(test_data['tot_id'].tolist())]

		return train_data[[self.user_col, self.item_col, self.interaction_col]], test_data[[self.user_col, self.item_col, self.interaction_col]]

	def negative_sampling(self, neg_data, neg_per_pos):
		"""
				Sample some number of negative data per positive
			data.
		"""

		neg_samples = pd.DataFrame()

		neg_groups = neg_data.groupby(self.user_col)

		for item in neg_groups:
			num_neg_samples = neg_per_pos * self.user2posnum_mapping[item[0]]
			if num_neg_samples <= len(item[1]):
				neg_samples = pd.concat([neg_samples, item[1].sample(num_neg_samples)], ignore_index = True)
			else:
				neg_samples = pd.concat([neg_samples, item[1].sample(num_neg_samples, replace = True)], ignore_index = True)

		# for item in neg_groups:
		# 	neg_samples = pd.concat([neg_samples, item[1].sample(neg_per_pos * self.user2posnum_mapping[item[0]], replace = True)], ignore_index = True)

		neg_samples = neg_samples.drop_duplicates()

		return neg_samples[[self.user_col, self.item_col, self.interaction_col]]

	def negative_sampling_test(self, neg_data, neg_per_pos_test):
		"""
				Sample some number of negative data per positive
			data.
		"""

		neg_samples_test = pd.DataFrame()

		neg_groups = neg_data.groupby(self.user_col)

		for item in neg_groups:
			if neg_per_pos_test <= len(item[1]):
				neg_samples_test = pd.concat([neg_samples_test, item[1].sample(neg_per_pos_test)], ignore_index = True)
			else:
				neg_samples_test = pd.concat([neg_samples_test, item[1].sample(neg_per_pos_test, replace = True)], ignore_index = True)

		# for item in neg_groups:
		# 	neg_samples = pd.concat([neg_samples, item[1].sample(neg_per_pos_test, replace = True)], ignore_index = True)

		neg_samples_test = neg_samples_test.drop_duplicates()

		return neg_samples_test[[self.user_col, self.item_col, self.interaction_col]]


class ImRecomDatasetTrain(torch.utils.data.Dataset):
	def __init__(self, X, y):
		super().__init__()
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return (torch.tensor(self.X.iloc[idx], dtype = torch.int64), torch.tensor(self.y.iloc[idx], dtype = torch.float32))


class ImRecomDatasetTest(torch.utils.data.Dataset):
	def __init__(self, X, y, user_col, item_col, interaction_col):
		super().__init__()
		self.user_col = user_col 
		self.item_col = item_col
		self.interaction_col = interaction_col
		self.user_groups = pd.concat([X, y], axis = 1)

		self.user_groups = list(self.user_groups.groupby(user_col))

	def __len__(self):
		return len(self.user_groups)

	def __getitem__(self, idx):
		return (torch.tensor(np.array(self.user_groups[idx][1][[self.user_col, self.item_col]], dtype = np.int64), dtype = torch.int64), torch.tensor(np.array(self.user_groups[idx][1][self.interaction_col], dtype = np.float), dtype = torch.float32))















