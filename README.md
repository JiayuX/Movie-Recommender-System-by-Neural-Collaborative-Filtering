# Movie-Recommender-System-by-Neural-Collaborative-Filtering
In this project, we build and train a movie recommender system based on neural collaborative filtering.

The dataset is from [Kaggle](https://www.kaggle.com/sherinclaudia/movielens). The data files need to be placed in the folder named 'data'.

We treat the rating from the users on the movies as a bianry implicit interaction, i.e. 0 for no interaction and 1 for observed interaction. The positive samples (1) is quite sparse as could be visualized by the image representing the user-movie interaction matrix.

<img src="https://raw.githubusercontent.com/JiayuX/Movie-Recommender-System-by-Neural-Collaborative-Filtering/main/matrix.png" width="350"/>

As the original paper described we performed leave-one-out and negative sampling to construct a training set and a validation set, which is used to train and evaluate the model performance. The performance is evaluated by the hit ratio (HR) and normalized discounted cumulative gain (NDCG). With 30 epochs of training, the model loss decreased to ~0.26 and HR and NDCG increased to ~0.39 and ~0.66, respectively.

<img src="https://raw.githubusercontent.com/JiayuX/Movie-Recommender-System-by-Neural-Collaborative-Filtering/main/history.png" width="900"/>

We compared the performance of the neural collaborative model to a baseline model. The baseline model makes recommendation purely based on the popularity of the items, i.e. the more popular a movie is the more probable it is regarded to be interacted by the users. The trained neural collaborative filtering model outperforms the baseline model 29.29% in HR and 32.62% in NDCG!

<img src="https://raw.githubusercontent.com/JiayuX/Movie-Recommender-System-by-Neural-Collaborative-Filtering/main/comp.png" width="600"/>

Due to the limited computational resources, we stopped the training at the 30th epoch and didn't tune the hyper-parameters. Although at around the 30th epoch the training curve and metric curves tend to saturate, there has not been a clear sign of overfitting yet. Therefore, training for more epochs may give a better result. Also, if hyper-parameters tuning is performed, the result may be further improved.
