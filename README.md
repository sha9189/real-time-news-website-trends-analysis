# News-Article-Classifier
A Machine Learning Based App

The app has been deployed here.

This app provides a dashboard where you can select any one of these news websites - India Today, Hindustan Times, Economic Times and The Guardian.
Once you select the option to Scrape, the application goes into the website and scraps out the title and links to all the articles. 

Then, using the links, it vists all the individual article pages and scraps out the articles. These articles are passed through a machine learning model to predict the category of each 
article. I used these 5 categories for classification - **Business**, **Politics**, **Entertainment**, **Sports** and **Tech**. The articles not 
belonging to any of these categories are labelled as Other. 

The dashboard uses this data to present insights like the number 
of articles posted in each category and the percentage of the webpage dedicated to each website.

[BBC News Raw Dataset](http://mlg.ucd.ie/datasets/bbc.html) which contains 2225 articles from 2004-2005 was used to train the model.


