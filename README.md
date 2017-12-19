# Projects
A collection of personal data visualization, data science, and machine learning projects

## About Me

I'm an engineer with a master's degree in applied mathematics and am currently pursuing certificates in data science 
and machine learning. I wanted to have a place to share some of the projects I've created while learning and exploring
topics in these fields.

## Documents in this Repository

**MontyHall_Simulation.pdf**

This project was a part of the second quarter of my data science certificate program. It simulates the famous "Monty Hall" problem, originally featured on the television game show program "Let's Make a Deal". In the game, a contestant is told that a prize exists behind one of three doors, and behind the other two doors are goats. The contestant picks a door, and one of the other two doors is then opened by the host revealing one of the goats. At this point, the contestant is allowed to switch their choice to the remaining door, or stick with their original choice. The optimal strategy is somewhat counterintuitive without evidence, so the writeup performs a simulation to determine the probability of winning the prize for each strategy (switching vs. staying) and summarizes the results.

**MovieSentimentNaiveBayes.ipynb**

This was the first machine learning project I completed using Python. I have used Python for over 7 years in my career at Boeing, but never before for this type of task. I wanted to created something from scratch, and Naive Bayes classifiers seemed like a great place to start because they are relatively easy to construct and train. This project uses sentiment analysis to attempt to classify movie reviews as being either positive or negative. Both multinomial and Bernoulli likelihood models were used. Confusion matrices, accuracy, sensitivity and specificity are used to evaluate the models, along with the overall runtime.

**Wine_Exploration.pdf**

This was my final project for the second quarter of my data science certificate program. I have always enjoyed wine, and was curious as to whether a country could claim to make the best wine in the world. I found a database of over 150,000 reviews from Wine Enthusiast magazine and set out to see if a statistically significant difference existed between different countries in terms of the average score awarded to a bottle of wine, and the price per bottle. During the exploratory phase, bar charts, boxplots, and scatterplots are created to analyze the data. In addition, correlation coefficients and linear regression models are used to determine the strength and dependence between the price and score for wines from each country. Various techniques are used to compare the wines from each country, including ANOVA, Tukey's HSD, and bootstrap methods. Finally, statistical power analysis is used to ensure that the countries being compared have enough samples in order to detect the desired difference in price or score at the desired significance and power levels.

**NYC_Taxi_Capstone_Project.pdf**

This was the capstone project for my data science certificate program, which I worked on with one other student. We looked at trip and fare data for ~1.7 million taxi rides in the greater New York City area, and were tasked with proposing a business question that could be answered using the data set. We wanted to focus on something that would bring value to a business, and set about trying to predict the length of a ride (in seconds) given only knowledge that would be known to the driver at the start of a trip. Project highlights include: Using k-means clustering to group rides by pickup location in order to model traffic, pulling in weather data from NOAA to add the effects of rain/snow on the length of a ride, and using hyperparameter tuning via 10-fold cross validation to select an appropriate machine learning model. Ultimately, a random forest regression model was able to predict rides with a validation RMSE of 4 minutes (the median ride length was 10 minutes), and a random forest classification model was able to predict trips and short, medium, or long with nearly 80% validation accuracy.
