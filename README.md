# Projects
A collection of personal data visualization, data science, and machine learning projects

## About Me

I'm an engineer with a master's degree in applied mathematics and certificates in data science and machine learning. I wanted to have a place to share some of the projects I've created while learning and exploring topics in these fields.

## Projects in this Repository

**MusicGenreLDA_DeVore.pdf**

This was the assignment from graduate school that introduced me to the world of machine learning. The objective of the assignment was to use several powerful mathematical techniques from the fields of signal processing (time-frequency analysis), linear algrbra (Singular Value Decomposition) and statistics (Linear Discriminant Analysis) to build an algorithm capable of accurately identifying specific genres of music.

**MontyHall_Simulation.pdf**

This project was a part of the second quarter of my data science certificate program. It simulates the famous "Monty Hall" problem, originally featured on the television game show program "Let's Make a Deal". In the game, a contestant is told that a prize exists behind one of three doors, and behind the other two doors are goats. The contestant picks a door, and one of the other two doors is then opened by the host revealing one of the goats. At this point, the contestant is allowed to switch their choice to the remaining door, or stick with their original choice. The optimal strategy is somewhat counterintuitive without evidence, so the writeup performs a simulation to determine the probability of winning the prize for each strategy (switching vs. staying) and summarizes the results.

**MovieSentimentNaiveBayes.ipynb**

This was the first machine learning project I completed using Python. I have used Python for over 7 years in my career at Boeing, but never before for this type of task. I wanted to created something from scratch, and Naive Bayes classifiers seemed like a great place to start because they are relatively easy to construct and train. This project uses sentiment analysis to attempt to classify 12,000 movie reviews as being either positive or negative. Both multinomial and Bernoulli likelihood models were used. Confusion matrices, accuracy, sensitivity and specificity are used to evaluate the models, along with the overall runtime. Both models achieved an accuracy of over 80%. Typically, it is thought that humans can only agree on the sentiment of a review about 80% of the time.

**Wine_Exploration.pdf**

This was my final project for the second quarter of my data science certificate program. I have always enjoyed wine, and was curious as to whether a country could claim to make the best wine in the world. I found a database of over 150,000 reviews from Wine Enthusiast magazine and set out to see if a statistically significant difference existed between different countries in terms of the average score awarded to a bottle of wine, and the price per bottle. During the exploratory phase, bar charts, boxplots, and scatterplots are created to analyze the data. In addition, correlation coefficients and linear regression models are used to determine the strength and dependence between the price and score for wines from each country. Various techniques are used to compare the wines from each country, including ANOVA, Tukey's HSD, and bootstrap methods. Finally, statistical power analysis is used to ensure that the countries being compared have enough samples in order to detect the desired difference in price or score at the desired significance and power levels.

**NYC_Taxi_Capstone_Project.pdf**

This was the capstone project for my data science certificate program, which I worked on with one other student. We looked at trip and fare data for ~1.7 million taxi rides in the greater New York City area, and were tasked with proposing a business question that could be answered using the data set. We wanted to focus on something that would bring value to a business, and set about trying to predict the length of a ride (in seconds) given only information that would be known to the driver at the start of a trip. Project highlights include: Using k-means clustering to group rides by pickup location in order to model traffic, pulling in weather data from NOAA to add the effects of rain/snow on the length of a ride, and using hyperparameter tuning via 10-fold cross validation to select an appropriate machine learning model. Ultimately, a random forest regression model is able to predict the length of a ride with a validation RMSE of 4 minutes (the median trip length is 10 minutes), and a random forest classification model is able to classify trips as short, medium, or long with nearly 80% validation accuracy.

**MarathonScraping.pdf**

This was my first serious attempt at web scraping, a topic that I'd known about for a while, but was honestly intimidated by. I finally decided to take the plunge, and use web scraping to look at the top 100 finishers in the NYC marathon since the year 2000 using R. I used both the *rvest* and *RSelenium* packages, which made what could have been a very frustrating experience relatively painless. Overall, I found that the top finishing times haven't improved much in the last 17 years, and that the men's times have a smaller standard deviation relative to the mean compared to the women's. Also, the distribution of finishing times is always left skewed, with the trend only getting more pronounced over time. In other words, although the winning times aren't improving, the bulk of the top 100 are consistently finishing in a tighter pack. This trend is especially prevalent in the women's results.

### Selected Assignments From Data Science Certificate Program

**AutoPrices_DeVore.pdf**

This assignment explores some of the basic concepts associated with hypothesis testing using an automobile data set. Tests of normality are applied to the data of interest (price), and significance tests are conducted on the price when stratified by binary and multivariate categories such as fuel type, aspiration type, or style.

**AutoPrices_Bootstrap_DeVore.pdf**

This assignment explores some of the basic concepts associated with hypothesis testing using an automobile data set, this type using the bootstrap. The results are compared with classical parametric methods (t-test, ANOVA, Tukey's HSD), and discrepancies are discussed.

**Bayesian_Auto_DeVore.pdf**

This assignment continues the with the automobile price data set, this time using Bayesian analysis to perform hypothesis testing. Starting with a prior distribution of automobile price, likelihood functions are computed as Gaussian distributions with mean and standard deviation from price when stratified by binary or multivariate categories. Once posterior distributions are computed, they are compared using 0.95 credible intervals to determine if a significant relationship exists on the price when stratified by a particular category.

**Time_Series_DeVore.pdf**

This assignment explores the topic of time series analysis, specifically to forecast ice cream sales over a year given 18 years of historical data. Topics include stationary time series, auto correlation and partial auto correlation, STL decomposition, and ARIMA forecasting.

### Selected Assignments From Machine Learning Certificate Program

**GregDeVore-L05-Wages.ipynb**

This assignment looks at City of Seattle employee wage data.  The goal was to build a model to predict hourly wage based on department, gender, and job title.  Gender was not included in the data set, so the 'gender' package in R was used to predict the most likely gender based on the employee's name.  One-hot encoding was used to transform the department feature, and Dracula counts were created to assign a probability to the job title based on its frequency within the data set.  Also, because there were a high number of features, a recursive feature elimination algorithm was run to find an optimal subset (based on the root mean squared error of a test data set).

**GregDeVore-L07-HomePrice.ipynb**

This assignment explored the idea of regularization using the popular housing price data set from Kaggle.  The first step was to impute missing values using both simple techniques and the 'MICE' package from R.  Also, heteroscedasticity in the residual versus fitted values plot from the linear regression model was resolved by taking the log of the response variable.  In addition, LASSO regularization was used as a means of feature selection, and its choice of the 'most important' features was investigated.  Finally, linear models were created using various forms of regularization (LASSO, Ridge, Elastic Net) and the validation set mean squared error of each was compared.

**GregDeVore-L08-WageAdmit.ipynb**

This assignment extends the standard linear regression model by investigating the use of polynomial fits, splines, and generalized additive models (using basis and smoothing splines).

**GregDeVore-L09-Bayes.ipynb**

This assignment explores a few topics in Bayesian analysis, including creating a simple Naive Bayes classifier and a Bayesian network using the 'bnlearn' package in R.  It also features a simple web scraping example using the 'rvest' package in R.

**GregDeVore-L10-PCA.ipynb**

This assignment using principal component analysis to explore the wine quality data set from Kaggle. In addition, principal components regression is compared to linear regression as a means of predicting wine quality.

**Lesson1_Input_and_Outputs_DeVore.ipynb**

This assignment introduces Python for machine learning, specifically exploratory data analysis and the standardization of features. It also includes building a classification tree from scratch using entropy and information gain to establish a set of rules. This tree is trained on a subset of the data and evaluated on the remaining observations.

**Lesson2_SVM_ANN_DeVore.ipynb**

This assignment explores the concept of Maximal Margin Classifiers through a simple handworked example. It also features a comparison between a linear support vector machine and multilayer perceptron for use in classifying the MNIST handwritten digits data set. Both models are tuned using a randomly sampled subset of the data, and confusion matrices are used to evaluate their accuracy.

**Lesson3_Decision_trees_Ensemble_methods_DeVore.ipynb**

This assignment explores the algorithm used to fit a regression tree. In addition, a random forest classifier is built from scratch using a series of single decision trees built on small random samples of the data set.

**Lesson4_Ensembles_Imputation_DeVore.ipynb**

This assignment explores imputing missing data in Python. Specifically, the precision and recall of a binary classifier is plotted as a function of the amount of missing data from a single feature. In addition, a custom imputation function is written to convert data types using regular expressions.

**Lesson5_Clustering_DeVore.ipynb**

This assignment begins with a handworked example illustrating how the k-means and hierarchical clustering algorithms are performed. In addition, the k-means algorithm was used to classify papers from a technical conference by topic. This also involved the generation of a term frequency, inverse document frequency (tf-idf) matrix so that distance measures could be computed between observations in the data set.

**Lesson9_Recommendation_Systems_DeVore.ipynb**

This assignment focuses on implementing a simple recommender system in Python. Specifically, memory and model based collaborative filters are used to predict ratings using the MovieLens data set. For the memory-based system, cosine similarity is used to create both user and item similarity matrices from a training data set, which are then used to predict ratings in a test data set of users and reviews. The model-based system used singular value decomposition to create a low rank approximation of the training data matrix, which was then used to predict reviews in the test data set. The root mean squared error (RMSE) of the test data set predictions was used to compare the methods.

### Selected Assignments From Deep Learning Course

**Lesson1a_Perceptron_Learning_DeVore.ipynb**

This assignment focuses on implementing a perceptron model with a step activation function from scratch. Although it is a simple model, proper understanding is essential to enable building of larger, more complex networks. The resulting model is used in a binary classification setting to classify species of flowers from the popular Iris data set. Particular attention was paid to the feedforward and weight updating portions of the process, to ensure understanding of how the network works 'under the hood'. The resulting decision boundary was plotted to visualize the separation between the classes. In addition, a classification task that was not linearly separable was attempted to show the limitations of the model.

**Lesson1b_Neural_Network_DeVore.ipynb**

This assignment introduced the keras library, in particular to solve the XOR problem. Since the data set is not linearly separable, this serves as an introduction to the concept of a hidden layer, and the power it brings to a classifier in terms of moving from linearly separable problems (no hidden layer), to nonlinearly separable problems (one or more hidden layers).

**Lesson2_DeVore_CustomNeuralNet.ipynb**

This assignment implements a custom neural network library from scratch. Input and dense layers with an arbitrary number of nodes are supported, along with random weight initialization and identity and sigmoid activation functions. The supported loss function is mean squared error, and accuracy is computed to support binary classification problems. Feed forward and back propagation steps are implemented by hand. Models can be constructed with multiple dense layers to support various architectures.

**RNN_Divisibility_DeVore.ipynb**

This assignment uses a recurrent neural network (RNN) to predict whether or not a number is divisible by three. The model encodes the number as a series of one-hot encoded sequences (one sequence per digit) and a single long-short term memory (LSTM) layer is added to enable the recurrent functionality. In addition, it is explored whether divisibility by other digits (such as 7 or 9) is possible using the same architecture, and theories as to why or why not are presented. Overall, the model is able to learn the 'rules' of divisibility by three with 100% accuracy, and numbers larger than those used in the training of the model are provided as a test.

**Translate_SeqToSeqRNN_DeVore.ipynb**

This assignment uses a sequence-to-sequence RNN to translate English sentences to French. Sentences are encoded using tokens corresponding to keys in a vocabulary dictionary, and fed to the network using a high dimensional embedding layer. A long-short term memory layer is used to enable the recurrent functionality, and softmax activation layers are used to predict the next word in the sequence. Overall, a 95% ‘perfect’ translation accuracy was obtained after training the network using 96,000 training sentences, and nearly 42,000 validation sentences.

**VAE_Assignment_DeVore.ipynb**

This assignment explores the concept of variational autoencoders (VAEs) using image data in keras. The idea behind a VAE (based on images), is that the network learns the probability distribution of the latent space corresponding to the training images. It can then sample from that space to generate new images. In particular, during training, an input image is passed through an encoder, which contains a series of convolutional layers. At the end, two outputs are generated, corresponding to the mean and variance of a Gaussian distribution which represents the latent space that the image was sampled from. A random sample is taken from this distribution, which is passed through a decoder consisting of a series of transpose convolution layers, until an output image is generated. There are two loss functions for the network, one to ensure the output image resembles the input image (a requirement for autoencoders), and another to ensure the latent space resembles a Gaussian distribution. One of the most interesting properties of the latent space is that it is continuous, and moving through the space results in smooth changes to the generated photos (such as changing the color of clothing, length of hair, or gender of the subject).

**GAN_DeVore.ipynb**

This assignment explores the concept of generative adversarial networks (GANs) using image data in keras. The idea behind a GAN (based on images) is to train a generator network to try and fool a descriminator network. The discriminator is trained to differentiate between real images (from a training set) and fake images (from the generator). The generator samples from random noise and outputs a fake image (labeled as real) in order to try and fool the generator into thinking it's real. Over time, the generator gets better at producing fake images, and the discriminator gets better at spotting them. These types of networks are notoriously hard to train, as there is no global minima to achieve, rather the two networks come to an equilibrium with one another. This particular example used the MNIST hand written digit database, and was able to generate very convincing fake images with only a few epochs of training.

**DQN_DeVore.ipynb**

This assignment explores the concept of deep reinforcement learning (DRL) using deep q-networks (DQNs) in keras. In particular, the network is trained to play the classic 'CartPole' game, where a moveable cart tries to keep a vertical pole upright. The network works by approximating the q-function using q-learning (a trial and error based approach). An epsilon greedy approach is used for training, where early actions taken by the agent are random (large value of epsilon). Over time, the value of epsilon decreases, and the DQN itself is used to predict the best action to take given the current state. Also, memory replay is used to train the agent (memories consists of actions taken at a given state, the reward received, and the next state achieved). Overall, the network was able to consistently keep the pole upright for at least 100 seconds after only 40 training epochs.
