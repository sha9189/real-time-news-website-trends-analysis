# News-Article-Classifier
A Machine Learning Based App

This is an end-to-end machine learning project. It involves the use of various skills like **text preprocessing**, **exploratory data analysis**, **model training**, **web scraping** and **dashboard creation**. 

This app starts with a dashboard where you can select any one of these news websites - India Today, Hindustan Times, Economic Times and The Guardian.
Once you select the option to Scrape, the application goes into the website and scraps out the title and links to all the articles. 

Then, using the links, it vists all the individual article pages and scraps out the articles. These articles are passed through a machine learning model to predict the category of each 
article. I used these 5 categories for classification - **Business**, **Politics**, **Entertainment**, **Sports** and **Tech**. The articles not 
belonging to any of these categories are labelled as Other. 

[BBC News Raw Dataset](http://mlg.ucd.ie/datasets/bbc.html) which contains 2225 articles from 2004-2005 was used for training the model. After iterations on multiple models and hyperparameter tuning, the chosen model gave an impressive accuracy of **96%** on the test set!

The dashboard uses this data to **visualize** insights like the number 
of articles posted in each category and the percentage of the webpage dedicated to each website.

Here is a snapshot of the application: 
![Application Screenshot](/Application%20Screenshot/Application%20Screenshot.png)

For any suggestions, comments or anything else, you can find me at my [mail](shailesh1121998@gmail.com).
