#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System 

# In[1]:


# Libraries Required
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print('Libraries Imported')


# In[2]:


#Loading the data
df = pd.read_csv('/Users/RK/Desktop/Rohit_Work/Semester_3/Web Mining/Module 6/Movie_Recommendation/ml-latest-small/ratings.csv')


# In[3]:


df.head()


# In[4]:


#Loading the data
movie_titles = pd.read_csv('/Users/RK/Desktop/Rohit_Work/Semester_3/Web Mining/Module 6/Movie_Recommendation/ml-latest-small/movies.csv')


# In[5]:


movie_titles.head()


# In[6]:


#Merging both the dataset 
df = pd.merge(df, movie_titles, how='inner', left_on=['movieId'], right_on=['movieId'])


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.describe() 


# ###### Let’s now create a data frame with the average rating for each movie and the number of ratings.

# In[10]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# In[11]:


ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings.head()


# ###### Let’s now plot a Histogram using pandas plotting functionality to visualize the distribution of the ratings.

# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ratings['rating'].hist(bins=5)


# ###### Let’s now plot a Histogram using pandas plotting functionality to visualize the distribution of the  number of ratings.

# In[13]:


ratings['number_of_ratings'].hist(bins=10)


# ###### Determining relationship between the rating of a movie and the number of ratings. 

# In[14]:


import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)


# In[15]:


# Create a movie matrix. 
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()


# In[16]:


# Determing the top 20 most rated movies by the users
ratings.sort_values('number_of_ratings', ascending=False).head(10)


# ###### create a data frame with the ratings of these movies from our movie_matrix.

# In[17]:


fg_user_rating = movie_matrix['Forrest Gump (1994)']
jp_user_rating = movie_matrix['Jurassic Park (1993)']
matrix_user_rating = movie_matrix['Matrix, The (1999)']


# In[18]:


fg_user_rating.head()


# In[19]:


jp_user_rating.head()


# In[20]:


matrix_user_rating.head()


# ###### Get the correlation between each movie's rating and rating of Forest Gump and Jurassic Park

# In[21]:


similar_to_fg=movie_matrix.corrwith(fg_user_rating)


# In[22]:


similar_to_fg.head()


# In[23]:


similar_to_jp = movie_matrix.corrwith(jp_user_rating)


# In[24]:


similar_to_jp.head()


# In[25]:


similar_to_matrix = movie_matrix.corrwith(matrix_user_rating)
similar_to_matrix.head()


# ###### Drop all the null values for both the data set 

# In[37]:


corr_jp = pd.DataFrame(similar_to_jp, columns=['correlation'])
corr_jp.dropna(inplace=True)
corr_jp.head()


# In[38]:


corr_fg = pd.DataFrame(similar_to_fg, columns=['correlation'])
corr_fg.dropna(inplace=True)
corr_fg.head()


# In[39]:


corr_matrix = pd.DataFrame(similar_to_matrix, columns=['correlation'])
corr_matrix.dropna(inplace=True)
corr_matrix.head()


# ###### Lets join the two data frame and obtaina a new data frame which indicates the number of rating and correlation of our chosen movie with other movies

# In[40]:


corr_fg = corr_fg.join(ratings['number_of_ratings'])
corr_fg.head()


# In[41]:


corr_jp = corr_jp.join(ratings['number_of_ratings'])
corr_jp.head()


# In[44]:


corr_matrix = corr_matrix.join(ratings['number_of_ratings'])
corr_matrix.head()


# ###### Determine the top 100 movies where more than 50 users have rated the movie.

# In[45]:


corr_fg[corr_fg['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10) #forest Gump


# In[43]:


corr_jp[corr_jp['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10) #Jurassic Park


# In[46]:


corr_matrix[corr_matrix['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10) #matrix


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




