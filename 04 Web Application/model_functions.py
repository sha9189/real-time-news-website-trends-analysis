import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from nltk.corpus import wordnet
import nltk
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time


#Import Done
#############################################################################################################

#All the scrappers here

#Scrapper for Hindustan Times starts
def scrap_ht():
    t0 = time.time()
    df_features = pd.DataFrame(columns = ["title", "link", "Content"])
    
    #Coverpage operations
    url = "https://www.hindustantimes.com/top-news/"
    coverpage_request = requests.get(url)
    coverpage = coverpage_request.content
    coverpage_soup = BeautifulSoup(coverpage, "html5lib")

    #Extracting the big stories of the coverpage
    #Using custom fuction to avoid capturing <div class= "media-heading headingfour"
    bignews = coverpage_soup.find_all(lambda tag: tag.name == "div"
                                     and tag.get("class") == ["media-heading"])
    links = [item.find("a")["href"] for item in bignews]
    titles = [item.find("a").get_text() for item in bignews]
    #Scrapping individual articles starts here
    # The philosophy is we fetch links one by one and only the links that can scraped and the corresponding titls will be added
    #in the below lists
    final_articles = []
    final_links = []
    final_titles = []

    session = requests.session()

    for n in np.arange(len(links)):
        if "/liveupdates.hindustantimes.com/" in links[n]:
            continue
        time.sleep(0.25)
        try:
            article = session.get(links[n])
            article_content = article.content
            soup_article = BeautifulSoup(article_content, "html5lib")
            body = soup_article.find("div", class_ = "storyDetail")
            final_article = " ".join([p.get_text() for p in body.find_all("p")])
            final_articles.append(final_article)
            final_titles.append(titles[n])
            final_links.append(links[n])
            print(n, "done")
        except:
            print(n, "failed")
    
    #Adding df_bignews articles into the main dataframe df_features
    df_bignews = pd.DataFrame({"title" : final_titles, "link" : final_links, "Content" : final_articles})
    df_features = df_features.append(df_bignews, ignore_index = True)
    
    #Scraping rest of the articles on the coverpage
    othernews = coverpage_soup.find_all("a", class_ = "wclink2")
    
    links = [item["href"] for item in othernews]
    titles = [item.get_text() for item in othernews]    
    

    # The philosophy is we fetch links one by one and only the links that can scraped and the corresponding titls will be added
    #in the below lists
    final_articles = []
    final_links = []
    final_titles = []

    for n in np.arange(len(links)):
        #Accountinng for non-scrapable links
        if "/liveupdates.hindustantimes.com/" in links[n]:
            continue
        time.sleep(0.25)
        try:
            article = session.get(links[n])
            article_content = article.content
            soup_article = BeautifulSoup(article_content, "html5lib")
            body = soup_article.find("div", class_ = "storyDetail")
            final_article = " ".join([p.get_text() for p in body.find_all("p")])
            final_articles.append(final_article)
            final_titles.append(titles[n])
            final_links.append(links[n])
            print(n, "done")
        except:
            print(n, "failed")
    
    #Adding these articles to mail dataframe df_features
    df_othernews = pd.DataFrame({"title" : final_titles, "link" : final_links, "Content" : final_articles})
    df_features = df_features.append(df_othernews, ignore_index=True)
    print("time taken for scrapping: {}".format(time.time()-t0))
    return df_features
    
#Scrapper for Hindustan Times Ends
#####################################

#Scrapper for India Today Starts
def scrap_india_today():
    t0 = time.time()
    df_features = pd.DataFrame(columns = ["title", "link", "Content"])
    url = "https://www.indiatoday.in/news.html"
    coverpage_request = requests.get(url)
    if coverpage_request.status_code != 200:
         sys.exit("Coverpage request couldn't be finished")
    coverpage = coverpage_request.content
    coverpage_soup = BeautifulSoup(coverpage, "html5lib")
    #Extracting the news section of the page
    coverpage_news = coverpage_soup.find_all("div", attrs= {"data-id" : "news-section"})
    #Iterating through each news section
    session = requests.session()
    all_titles = []
    all_links = []
    all_content = []
    i=0
    for news_section in coverpage_news:


        #The structure of the page is such that the first headline has <h3> tag and following few headlines have <p> tag

        #First extracting the <h3> tag
        title = ""
        link = ""
        final_article = ""
        title = news_section.find("h3")["title"]
        link = "https://www.indiatoday.in" + news_section.find("h3").find("a")["href"]
        if ("/video/" not in link) and ("/photo/" not in link):
            #Introducing delay to be polite with the server
            #t0 = time.time()
            try:
                final_article = link_to_content(link, session)
                df_features = push_to_df(df_features, title=title, link=link, content=final_article)
                print("{} done".format(i))
                time.sleep(0.25)
            except:
                print("{} failed".format(i))
            #response_delay = time.time() - t0
            #time.sleep(5*response_delay)
            #Pushing to dataframe
        i+=1

        #Next extracting the <p> tags
        headlines = news_section.find_all("p")
        for headline in headlines:
            title = ""
            link = ""
            final_article = ""
            title = headline["title"]
            link = "https://www.indiatoday.in" + headline.find("a")["href"]
            if ("/video/" not in link) and ("/photo/" not in link):
                try:
                    final_article = link_to_content(link, session)
                    df_features = push_to_df(df_features, title=title, link=link, content=final_article)
                    print("{} done".format(i))
                    time.sleep(0.25)
                #Introducing delay to be polite with the server
                #t0 = time.time()
                except:
                    print("{} failed".format(i))
                #response_delay = time.time() - t0
                #time.sleep(5*response_delay)
                #Pushing to dataframe
                i+=1
    print("time taken for scrapping: {}".format(time.time()-t0))
    return df_features
    
def link_to_content(link, session):
    article_page = session.get(link)
    article_content = article_page.content
    soup_article = BeautifulSoup(article_content, "html5lib")
    articleBody = soup_article.find("div", attrs = {"itemprop" : "articleBody"})
    #searching for html with structure <p> text </p> only
    list_paras = articleBody.find_all("p")
    final_article = ""
    for p in list_paras:
        #To remove all ads with structure like <p><a>.........</a><p>
        if len(p.find_all())!=0:
            continue
        final_article += p.get_text()
    return final_article

def push_to_df(df, title, link, content):
    df2 = pd.DataFrame({"title": [title], "link" : [link], "Content" : [content]})
    df = df.append(df2, ignore_index = True)
    return df
    
#Scrapper for India Today Ends    
##########################################

#Scrapper for Economic Times Starts
def scrap_econotic_times():
    t0 = time.time()
    url = "https://economictimes.indiatimes.com/"
    coverpage_request = requests.get(url)
    coverpage = coverpage_request.content
    coverpage_soup = BeautifulSoup(coverpage, "html5lib")
    #Extracting the top news section of the coverpage
    topnews = coverpage_soup.find("li", attrs= {"data-ga-action" : "Widget Top News"})
    li = [li for li in topnews.find_all("li") if li.attrs == {} ]
    links = [url + item.find("a")["href"] for item in li]
    titles = [item.get_text() for item in li]
    session = requests.session()
    # The philosophy is we fetch links one by one and only the links that can scraped and the corresponding titls will be added
    #in the below lists
    final_articles = []
    final_links = []
    final_titles = []
    #crawling through all the links
    for n in np.arange(len(links)):
        try:
            article = session.get(links[n])
            article_content = article.content
            soup_article = BeautifulSoup(article_content, "html5lib")
            body = soup_article.find("div", class_ = "Normal")
            final_article = body.get_text()
            final_articles.append(final_article)
            final_titles.append(titles[n])
            final_links.append(links[n])
            print(n, "done")
            time.sleep(0.25)
        except:
            print(n, "failed")
    df_features = pd.DataFrame({"title" : final_titles, "link" : final_links, "Content" : final_articles})
    print("time taken for scrapping: {}".format(time.time()-t0))
    return df_features
#Scrapper for Economic Times Ends
###########################################

#Scrapper for The Guardian starts here




def scrap_Guardian():
    before = time.time()
    url = "https://www.theguardian.com/uk"

    r1 = requests.get(url)


    coverpage = r1.content

    soup1 = BeautifulSoup(coverpage, "html5lib")
    coverpage_news = soup1.find_all('h3', class_='fc-item__title')
    number_of_articles = len(coverpage_news)
    # Empty lists for content, links and titles
    session = requests.session()
    news_contents = []
    list_links = []
    list_titles = []
    for n in np.arange(0, number_of_articles):

        if ("live" in coverpage_news[n].find("a")["href"]) or ("ng-interactive" in coverpage_news[n].find("a")["href"]) or ("audio" in coverpage_news[n].find("a")["href"]) or ("video" in coverpage_news[n].find("a")["href"]) or ("gallery" in coverpage_news[n].find("a")["href"]) or ("picture" in coverpage_news[n].find("a")["href"]):
            print(n, "skipped")
            continue
        if n in [0, 2, 4, 6]:
            print(n, "skipped duplicate")
            continue
        #Remove this to be polite to the website
        #time.sleep(0.25)
        #Getting link of article
        time.sleep(0.25)
        link = coverpage_news[n].find("a")["href"]

        #Getting title of Article
        title = coverpage_news[n].find("a").get_text()

        #Reading the content
        #t1 = time.time()

        article = session.get(link)
        #response_delay = time.time() - t1
        #time.sleep(max(5*response_delay, 2))

        article_content = article.content
        soup_article = BeautifulSoup(article_content, "html5lib")
        body = soup_article.find("div", class_ = "content__article-body from-content-api js-article__body")
        #To handle being blocked by server
        try:
            final_article = " ".join([p.get_text() for p in body.find_all("p")])
            list_links.append(link)
            list_titles.append(title)
            news_contents.append(final_article)
            print(n, "done")
        except:
            print(n, "failed")
    print("time taken: {}".format(time.time()-before))
    df_articles = pd.DataFrame({"title" : list_titles, "link" : list_links,"Content" : news_contents, })
    #Since 1st 2 article links get scrapped twice and he 
    return df_articles

    
#Scrapper for the Guardian Ends
#############################################

###############################################################################
#Model Functions start here

#Funtion to get word type for lemmitization
def get_wordnet_pos(word):
    #Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
             "N": wordnet.NOUN,
             "V": wordnet.VERB,
             "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#Importing the tfidf object for creating features out of scrapped articles    
folder_path = "D://Projects//News Article classifier//01 Dataset Creation//Pickles//"
filename = "tfidf.pickle"
with open(folder_path + filename, "rb") as file:
    tfidf = pickle.load(file)
    
#List of stop words extracted from https://gist.github.com/sebleier/554280
stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

#Function to convert article text to tfidf features also called feature engineering
def create_features_from_content(df):    
    
    #Initial time
    t0 = time.time()
    #Convert everything to lowercase
    df["Content_parsed_1"] = df["Content"].str.lower()
    
    #Possesive Pronouns
    df["Content_parsed_2"] = df["Content_parsed_1"].str.replace("'s", "").str.replace("’s", "")

    #Removing stop words
    df["Content_parsed_3"] = df['Content_parsed_2']
    df["Content_parsed_3"] = df["Content_parsed_3"].str.replace("’", "'")
    
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_parsed_3'] = df['Content_parsed_3'].str.replace(regex_stopword, '')
    
    #Removing punctuation signs
    df["Content_parsed_4"] = df["Content_parsed_3"]
    punct_signs = [ "'", '"' , '“' , '”' , "\n" , "(" , ")" , "," , "." , "?", "-"]
    for sign in punct_signs:
        df["Content_parsed_4"] = df["Content_parsed_4"].str.replace(sign, " ")

    #Lemmitization
    lemmatizer = WordNetLemmatizer()
    all_articles = df["Content_parsed_4"]
    all_articles_l = []
    for article in all_articles:
        article_words = article.split()
        article_words_l = []
        for article_word in article_words:
            article_words_l.append(lemmatizer.lemmatize(article_word, pos = get_wordnet_pos(article_word)))
        article_l = " ".join(article_words_l)
        all_articles_l.append(article_l)
    df["Content_parsed_5"] = all_articles_l

    #Deleting all intermediate columns and retaining only the last column after processing
    df = df['Content_parsed_5']
    df = df.rename(columns = {"Content_parsed_5" : "Content_parsed"})
    
    #Transforming text into feautures
    df = tfidf.transform(df)
    print("Time taken for tfidf:{}".format(time.time()-t0))
    return df
    
#Using model to predict from features

#Importing trained Model here
models_folder = "D://Projects//News Article classifier//02 Model Training//Models//"
with open(models_folder + "best_rfc.pickle", "rb") as file:
    model = pickle.load(file)

def get_category_name(category_code):
    category_names = {0 : "Business",
                 1 : "Entertainment",
                 2 : "Politics",
                 3 : "Sports",
                 4 : "Tech",
                 5 : "Other"}
    for id, name in category_names.items():
        if id == category_code:
            return name

def predict_from_features(features):
    #features = create_features_from_text(text)
    model_prediction = model.predict(features)
    prediction_prob = model.predict_proba(features).max(axis = 1)
    prediction = []
    for prob, cat in zip(prediction_prob, model_prediction):
        if prob>0.45:
            prediction.append(cat)
        else:
            prediction.append(5)
    categories = [get_category_name(x) for x in prediction]
    return categories

def complete_df(df, categories):
    df['Prediction'] = categories
    return df[['title', 'link', 'Content', 'Prediction']]

#Model Function Ends Here
#########################################

if __name__ == "__main__":
    df_articles = scrap_ht()
    features = create_features_from_content(df_articles)
    categories = predict_from_features(features)
    df = complete_df(df_articles, categories)
    print(df["Prediction"])
    print("done")
    