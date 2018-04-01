# Projects
A collection of personal data visualization, data science, and machine learning projects

## About Me

I'm an engineer with a master's degree in applied mathematics and certificates in data science and machine learning (this one's still in progress). I wanted to have a place to share some of the projects I've created while learning and exploring topics in these fields.

## Projects in this Repository

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

This assignment explores some of the basic concepts associated with hypothesis testing using an autombile data set. Tests of normality are applied to the data of interest (price), and significance tests are conducted on the price when stratified by binary and multivariate categories such as fuel type, aspiration type, or style.

**Bayesian_Auto_DeVore.pdf**

This assignment continues the with the automobile price data set, this time using Bayesian analysis to perform hypothesis testing. Starting with a prior distribution of automobile price, likelihood functions are computed as Gaussian distributions with mean and standard deviation from price when stratified by binary or multivariate categories. Once posterior distributions are computed, they are compared using 0.95 credible intervals to determine if a significant relationship exists on the price when stratified by a particular category.

**Time_Series_DeVore.pdf**

This assignment explores the topic of time series analysis, specifically to forecast ice cream sales over a year given 18 years of historical data. Topics include stationary time series, auto correlation and partial auto correlation, STL decomposition, and ARIMA forecasting.

### Selected Assignments From Machine Learning Certificate Program

**GregDeVore-L05-Wages.ipynb**

This assignment looks at City of Seattle employee wage data.  The goal was to build a model to predict hourly wage based on department, gender, and job title.  Gender was not included in the data set, so the 'gender' package in R was used to predict the most likely gender based on the employee's name.  One-hot encoding was used to transform the department feature, and Dracula counts were created to assign a probability to the job title based on its frequency within the data set.  Also, because there were a high number of features, a recursive feature elimination algorithm was run to find an optimal subset (based on the root mean squared error of a test data set).

**GregDeVore-L07-HomePrice.ipynb**

This assignment explored the popular housing price data set from Kaggle.  Highlights include
* Imputing missing values using simple techniques along with the 'MICE' package from R.
* Resolving heteroscedasticity in the residual versus fitted values plot from the linear regression model by taking the log of the response variable.
* Using LASSO regularization as a means of feature selection and commenting its choice of the 'most important' features.
* Create linear models using various forms of regularization (LASSO, Ridge, Elastic Net) and comparing the validation set mean squared error of each.
