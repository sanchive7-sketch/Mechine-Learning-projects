import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
try:
    ratings = pd.read_csv('recommender_dataset_200_users (1).csv', encoding='latin1')
except (UnicodeDecodeError, pd.errors.ParserError):
    ratings = pd.read_excel('recommender_dataset_200_users (1).csv')
ratings.set_index('User', inplace=True)
print(ratings.head())
cos_sim = cosine_similarity(ratings)
print(cos_sim[:2,:2])
pearson_sim = ratings.T.corr()
print(pearson_sim.iloc[:2,:2])
svd = TruncatedSVD(n_components=3)
reduced_matrix = svd.fit_transform(ratings)
print('shape', reduced_matrix.shape)
approx_matrix = svd.inverse_transform(reduced_matrix)
approx_df = pd.DataFrame(approx_matrix, index=ratings.index, columns=ratings.columns)
print(approx_df.loc['User4'].sort_values(ascending=False).head(3))
