import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import warnings
from pandas import DataFrame

sns.set()
warnings.filterwarnings('ignore')


class Prediction:
	def __init__(self):
		self.model_dict: dict = {}
		self.cat1_list = ['WOM', 'BEA', 'KID', 'ELE', 'MEN', 'HOM', 'VIN', 'OTH', 'HAN', 'SPO', 'NOI']
		self.cat_df: DataFrame = pd.DataFrame(columns=['cat1', 'cat2', 'cat3', 'brand_name'])

	# #####################################################################################################################
	def prediction(self, df_tr, df_ts):
		print(self.cat1_list)
		p_result = pd.DataFrame()

		for ct in self.cat1_list:
			print("start get_prediction_model for [ " + ct + " ]")
			start_time = time.time()
			temp_tr_i = df_tr[df_tr['cat1'] == ct]
			temp_ts_i = df_ts[df_ts['cat1'] == ct]

			if ct != 'HAN':
				# train brand_name not > 5 --> 'Others'
				brand_count = temp_tr_i.groupby('brand_name')['brand_name'].count()
				b_count_dict: dict = brand_count.to_dict()
				d = {}
				for b_count in b_count_dict.keys():
					if b_count_dict[b_count] > 5:
						d[b_count] = b_count
					else:
						d[b_count] = 'Others'
				temp_tr_i['brand_name'] = temp_tr_i['brand_name'].map(d)

				# test brand_name not in train --> 'Others'
				tr_b = set(list(temp_tr_i['brand_name']))
				ts_b = set(list(temp_ts_i['brand_name']))
				b_dict = {}
				for b in ts_b:
					if b in tr_b:
						b_dict[b] = b
					else:
						b_dict[b] = 'Others'

				b_dict['no_info'] = 'Others'
				temp_ts_i['brand_name'] = temp_ts_i['brand_name'].map(b_dict)

			if ct != 'NOI':
				ct2_count = temp_tr_i.groupby('cat2')['cat2'].count()
				ct2_count_dict: dict = ct2_count.to_dict()
				ct2_d = {}
				for ct2_count in ct2_count_dict.keys():
					if ct2_count_dict[ct2_count] > 5:
						ct2_d[ct2_count] = ct2_count
					else:
						ct2_d[ct2_count] = 'Other'
				temp_tr_i['cat2'] = temp_tr_i['cat2'].map(ct2_d)

				ct3_count = temp_tr_i.groupby('cat3')['cat3'].count()
				ct3_count_dict: dict = ct3_count.to_dict()
				ct3_d = {}
				for ct3_count in ct3_count_dict.keys():
					if ct3_count_dict[ct3_count] > 5:
						ct3_d[ct3_count] = ct3_count
					else:
						ct3_d[ct3_count] = 'Other'
				temp_tr_i['cat3'] = temp_tr_i['cat3'].map(ct3_d)

				tr_ct2 = set(list(temp_tr_i['cat2']))
				ts_ct2 = set(list(temp_ts_i['cat2']))
				ct2_dict = {}
				for ct2 in ts_ct2:
					if ct2 in tr_ct2:
						ct2_dict[ct2] = ct2
					else:
						ct2_dict[ct2] = 'Other'

				ct2_dict['no_info'] = 'Other'
				ct2_dict['Others'] = 'Other'
				temp_ts_i['cat2'] = temp_ts_i['cat2'].map(ct2_dict)

				tr_ct3 = set(list(temp_tr_i['cat3']))
				ts_ct3 = set(list(temp_ts_i['cat3']))
				ct3_dict = {}
				for ct3 in ts_ct3:
					if ct3 in tr_ct3:
						ct3_dict[ct3] = ct3
					else:
						ct3_dict[ct3] = 'Other'

				ct3_dict['no_info'] = 'Other'
				ct3_dict['Others'] = 'Other'
				temp_ts_i['cat3'] = temp_ts_i['cat3'].map(ct3_dict)

				print(sorted(set(list(temp_tr_i['brand_name']))))

				if ct == 'HAN':
					formula = 'logPrice ~ item_condition_id + shipping + cat2 + cat3'
					# vcf = {'item_condition_id': 'C(item_condition_id)', 'shipping': 'C(shipping)',
					#            'cat2': 'C(cat2)', 'cat3': 'C(cat3)'}
					# temp_tr_i["grp"] = temp_tr_i["item_condition_id"].astype(str) + temp_tr_i["shipping"].astype(str) \
					#                    + temp_tr_i["cat2"].astype(str) + temp_tr_i["cat3"].astype(str)

				else:
					formula = 'logPrice ~ item_condition_id + brand_name + shipping + cat2 + cat3'
					# vcf = {'item_condition_id': 'C(item_condition_id)', 'brand_name': 'C(brand_name)',
					#        'shipping': 'C(shipping)', 'cat2': 'C(cat2)', 'cat3': 'C(cat3)'}
					# temp_tr_i["grp"] = temp_tr_i["item_condition_id"].astype(str) + temp_tr_i["brand_name"].astype(str) \
					# 				+ temp_tr_i["shipping"].astype(str) + temp_tr_i["cat2"].astype(str) + temp_tr_i["cat3"].astype(str)

			else:
				formula = 'logPrice ~ item_condition_id + brand_name + shipping'
				# vcf = {'item_condition_id': 'C(item_condition_id)', 'brand_name': 'C(brand_name)',
				#            'shipping': 'C(shipping)'}
				# temp_tr_i["grp"] = temp_tr_i["item_condition_id"].astype(str) + temp_tr_i["brand_name"].astype(str)\
				#                    + temp_tr_i["shipping"].astype(str)

			md = sm.MixedLM.from_formula(formula, groups=temp_tr_i['cat1'], data=temp_tr_i)

			self.model_dict[ct] = md.fit()
			print(self.model_dict[ct].summary())
			print("get_prediction_model for [ " + ct + " ] -- %.6f sec --" % (time.time() - start_time))

			start_time = time.time()
			temp_ts_i["logPrice"] = self.model_dict[ct].predict(temp_ts_i)
			p_result = p_result.append(temp_ts_i)
			print(p_result)
			print("set_predicted_result for [ " + ct + " ] -- %.6f sec --" % (time.time() - start_time))

			self.model_dict[ct].save(".\\model\\longley_results_" + ct + ".pickle")

		self.cat_df[['cat1', 'cat2', 'cat3', 'brand_name']] = df_tr[['cat1', 'cat2', 'cat3', 'brand_name']]
		self.cat_df = self.cat_df.drop_duplicates()
		self.cat_df.to_csv(".\\cat_table\\cat_df.csv")

		def exp_fun(x):
			return np.exp(x) - 1

		p_result['predicted_price'] = p_result['logPrice'].apply(exp_fun)
		p_result.sort_values(by=[p_result.columns[0]])
		print("prediction finished")
		return p_result

	# #####################################################################################################################
	def predicted_one_value(self, values_dict: dict):
		print('Load model files')
		ct = values_dict['cat1']
		self.model_dict[ct] = sm.load('.\\model\\longley_results_' + ct + '.pickle')

		values_dict['test_id'] = 0
		values_list = [[values_dict['test_id'], values_dict['cat1'], values_dict['cat2'], values_dict['cat3'],
		     values_dict['brand_name'], values_dict['item_condition_id'], values_dict['shipping']]]

		parameters = pd.DataFrame(values_list, columns=['test_id', 'cat1', 'cat2', 'cat3', 'brand_name', 'item_condition_id', 'shipping'])
		# cat_df = pd.read_csv(".\\cat_table\\cat_df.csv")
		# self.tidy_up_test_data(parameters, cat_df)
		print(parameters)
		parameters['log_price'] = self.model_dict[ct].predict(parameters)
		result = float(parameters[parameters['test_id'] == 0]['log_price'])
		r_value = str(math.exp(result) - 1)
		print(r_value)
		return r_value

	# ##################################################################################################################
	def prediction_df(self, df_ts):
		print('Load model files')
		for ct in self.cat1_list:
			self.model_dict[ct] = sm.load('.\\model\\longley_results_' + ct + '.pickle')

		print(self.cat1_list)
		p_result = pd.DataFrame()

		for ct in self.cat1_list:
			print("start get_prediction_model for [ " + ct + " ]")
			temp_ts_i = df_ts[df_ts['cat1'] == ct]
			temp_ct_i = self.cat_df[self.cat_df['cat1'] == ct]

			if ct != 'HAN':
				# test brand_name not in train --> 'Others'
				ct_b = set(list(temp_ct_i['brand_name']))
				ts_b = set(list(temp_ts_i['brand_name']))
				b_dict = {}
				for b in ts_b:
					if b in ct_b:
						b_dict[b] = b
					else:
						b_dict[b] = 'Others'

				b_dict['no_info'] = 'Others'
				temp_ts_i['brand_name'] = temp_ts_i['brand_name'].map(b_dict)

			if ct != 'NOI':
				tr_ct2 = set(list(temp_ct_i['cat2']))
				ts_ct2 = set(list(temp_ts_i['cat2']))
				ct2_dict = {}
				for ct2 in ts_ct2:
					if ct2 in tr_ct2:
						ct2_dict[ct2] = ct2
					else:
						ct2_dict[ct2] = 'Other'

				ct2_dict['no_info'] = 'Other'
				ct2_dict['Others'] = 'Other'
				temp_ts_i['cat2'] = temp_ts_i['cat2'].map(ct2_dict)

				tr_ct3 = set(list(temp_ct_i['cat3']))
				ts_ct3 = set(list(temp_ts_i['cat3']))
				ct3_dict = {}
				for ct3 in ts_ct3:
					if ct3 in tr_ct3:
						ct3_dict[ct3] = ct3
					else:
						ct3_dict[ct3] = 'Other'

				ct3_dict['no_info'] = 'Other'
				ct3_dict['Others'] = 'Other'
				temp_ts_i['cat3'] = temp_ts_i['cat3'].map(ct3_dict)

			start_time = time.time()
			temp_ts_i["logPrice"] = self.model_dict[ct].predict(temp_ts_i)
			p_result = p_result.append(temp_ts_i)
			print(p_result)
			print("set_predicted_result for [ " + ct + " ] -- %.6f sec --" % (time.time() - start_time))

		def exp_fun(x):
			return math.exp(x) - 1

		p_result['predicted_price'] = p_result['logPrice'].apply(exp_fun)
		p_result.sort_values(by=[p_result.columns[0]])
		print("prediction finished")
		return p_result

	# ##################################################################################################################
	def tidy_up_train_data(self, df: DataFrame):
		# Exclude `item_description` for this analysis
		# Split category variable into five category levels --xx(and take three levels into data)xx--

		start_total_time = start_time = time.time()

		new = pd.DataFrame(df.category_name.astype(str).str.split('/').tolist(),
		                   columns=['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])

		for c in new:
			df[c] = new[c]

		print("Split category -- %.6f sec --" % (time.time() - start_time))

		# Exclude the original category name, item_description, cat4 and cat5 variable
		df = df.drop(['item_description', 'category_name', "cat4", "cat5"], axis=1)

		# Fill missing values of categories as 'no_info'
		start_time = time.time()
		df = df.fillna('no_info')
		print("Fill missing values -- %.6f sec --" % (time.time() - start_time))

		# Reduce the dimension of brand name
		# Replace brand name which are not in the pop_brand list as `Others'

		start_time = time.time()

		# sub 1
		brand_group = df.groupby('brand_name')
		brand_count = brand_group['brand_name'].count()
		brand_mean = brand_group['price'].mean()

		b_mean_dict: dict = brand_mean.to_dict()
		b_count_dict: dict = brand_count.to_dict()

		for b_count in b_count_dict.keys():
			if b_count_dict[b_count] < 100:
				del b_mean_dict[b_count]

		b_mean_dict = dict(sorted(b_mean_dict.items(), key=lambda item: item[1]))

		pop_brand = list(b_mean_dict.keys())[-150:]

		# sub 3
		d = {}
		sub_start_time = time.time()
		for b_count in b_count_dict.keys():
			if b_count in pop_brand:
				d[b_count] = b_count

		d['no_info'] = 'Others'
		df['brand_name'] = df['brand_name'].map(d)
		df['brand_name'] = df['brand_name'].fillna('Others')
		print("sub 3: .map({b_count: 'Others'} -- %.6f sec --" % (time.time() - sub_start_time))

		print("Replace brand name not in the pop_brand list as `Others' -- %.6f sec --" % (time.time() - start_time))

		# Mutation log transformed price as "logPrice"
		start_time = time.time()

		def log_price_create(price):
			return np.log(price + 1)

		df["logPrice"] = df.price.apply(log_price_create)
		print("Create log(Price) -> logPrice -- %.6f sec --" % (time.time() - start_time))

		# Change cat1 label
		start_time = time.time()
		cat_rename_dict = {'Women': 'WOM', 'Beauty': 'BEA', 'Kids': 'KID', 'Electronics': 'ELE', 'Men': 'MEN',
		                   'Home': 'HOM',
		                   'Vintage & Collectibles': 'VIN', 'Other': 'OTH', 'Handmade': 'HAN',
		                   'Sports & Outdoors': 'SPO',
		                   'nan': 'NOI'}
		df['cat1'] = df['cat1'].map(cat_rename_dict)
		print("Change cat1 label -- %.6f sec --" % (time.time() - start_time))

		print("tidy_up_data -- %.6f sec --" % (time.time() - start_total_time))

		return df

	# #################################################################################################################
	def tidy_up_test_data(self, df: DataFrame, tr_df):
		# Exclude `item_description` for this analysis
		# Split category variable into five category levels --xx(and take three levels into data)xx--

		start_total_time = start_time = time.time()

		new = pd.DataFrame(df.category_name.astype(str).str.split('/').tolist(),
		                   columns=['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])
		for c in new:
			df[c] = new[c]

		print("Split category -- %.6f sec --" % (time.time() - start_time))

		# Exclude the original category name, item_description, cat4 and cat5 variable
		df = df.drop(['item_description', 'category_name', "cat4", "cat5"], axis=1)

		# Fill missing values of categories as 'no_info'

		start_time = time.time()
		df = df.fillna('no_info')
		print("Fill missing values -- %.6f sec --" % (time.time() - start_time))

		# Reduce the dimension of brand name

		# Replace test brand name which are not in the train brand name list as `Others'
		start_time = time.time()

		b_dict: dict = {}

		pop_brand = set(list(tr_df['brand_name']))
		ts_df = set(list(df['brand_name']))
		for b in ts_df:
			if b not in pop_brand:
				b_dict[b] = 'Others'
			else:
				b_dict[b] = b

		b_dict['no_info'] = 'Others'
		b_dict['Other'] = 'Others'
		df['brand_name'] = df['brand_name'].map(b_dict)

		print("Replace brand name if not in pop_brand as `Others' -- %.6f sec --" % (time.time() - start_time))

		# Change cat1 label
		start_time = time.time()
		cat_rename_dict = {'Women': 'WOM', 'Beauty': 'BEA', 'Kids': 'KID', 'Electronics': 'ELE', 'Men': 'MEN',
		                   'Home': 'HOM',
		                   'Vintage & Collectibles': 'VIN', 'Other': 'OTH', 'Handmade': 'HAN',
		                   'Sports & Outdoors': 'SPO',
		                   'nan': 'NOI'}
		df['cat1'] = df['cat1'].map(cat_rename_dict)
		print("Change cat1 label -- %.6f sec --" % (time.time() - start_time))
		print("tidy_up_test_data -- %.6f sec --" % (time.time() - start_total_time))

		return df

# #####################################################################################################################
# Reading train and test datasets
# start_time1 = time.time()
#
# df_train: DataFrame = pd.read_csv(".\\input\\train.tsv", sep='\t')
# df_test: DataFrame = pd.read_csv(".\\input\\test.tsv", sep='\t')
#
# print("Reading train and test datasets -- %.4f sec --" % (time.time() - start_time1))
#
# p = Prediction()
# train_1 = p.tidy_up_train_data(df_train.head(10000))
# test_1 = p.tidy_up_test_data(df_test.head(6000), train_1)
#
# result = p.prediction(train_1, test_1)
#
# d = vcf = {'item_condition_id': '3', 'brand_name': 'Others', 'shipping': '1', 'cat1': 'MEN', 'cat2': 'Tops', 'cat3': 'T-shirts'}
# p.predicted_one_value(d)
#
# # print(result['price'])
# # result.to_csv(".\\output\\result_p.csv")
