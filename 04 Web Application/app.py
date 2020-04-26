import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from model_functions import scrap_ht, create_features_from_content, predict_from_features, complete_df, scrap_india_today, scrap_econotic_times, scrap_Guardian
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Adding only for testing purpose
#df = pd.read_csv("ht_predicted.csv")
#df_json = df.to_json(orient='split')

# Colors
colors = {
    'background': '#ECECEC',
    'text': '#696969',
    'titles': '#599ACF',
    'blocks': '#F7F7F7',
    'graph_background': '#F7F7F7',
    'banner': '#C3DCF2',
    'boxShadow': '#9D9D9D'
}

# Markdown text to be used later
markdown_text1 = '''
This application is capable of scraping all articles present in the cover page of top news websites in real-time. A machine learning model predicts if the article falls in any of these five categories - **Business**, **Entertainment**, **Sports**, **Tech** and **Politics**.

This data is used to get insights like - 
How many articles are present on the cover page in each category? or   
How much percentage of the cover page is devoted to each category?

Please enter which news website you would like to scrap and press the **Scrape** button.
'''


markdown_text2 = '''
Websites scraped for each category:   
1. [India Today](https://www.indiatoday.in/news.html)   
2. [Hindustan Times](https://www.hindustantimes.com/top-news/)   
3. [Economic Times](https://economictimes.indiatimes.com/) (Top news widget)   
4. [The Guardian](https://www.theguardian.com/uk)   

Created by Shailesh Mahto as part of an educational project
'''


app.layout = html.Div([
    # Space before title
    html.H1(' '
            , style={'padding': '10px'}
            ),
    # Title
    html.Div([
        html.H3("News Classification App"
                , style={"margin-bottom": "0px"}
                ),
        html.H6("A Machine Learning Based App")
    ],
        style={
            'textAlign': 'center'
            , 'color': colors['text']
            , 'backgroundColor': colors['background']
        }
        , className='banner'
    ),

    # Space after title - just as option iif I feel like adding a space there later
    html.H1([
        " "
    ], style={'padding': '0px'}),

    # Text Boxes
    html.Div([
        # Left half block
        html.Div([
            # Title of the div
            html.H6("What does this app do?"
                    , style={'color': colors["titles"]}
                    ),
            # Text in div
            html.Div(
                [dcc.Markdown(markdown_text1,
                    style={
                        'font-size': '12px'
                        , 'color': colors['text']
                    }
                )
                ]
                
                
            ),
            # Dropdown in div
            html.Div([
                dcc.Dropdown(
                    options=[
                        {'label': 'India Today', 'value': 'IT'}
                        , {'label': 'Hindustan Times', 'value': 'HT'}
                        , {'label': 'Economic Times', 'value': 'ET'}
                        , {'label': 'The Guardian', 'value': 'TG'}
                    ],
                    id='checklist'
                )
            ]
                , style={
                    'font-size': '12px',
                    'margin-top': '25px'
                }),
            # Submit button next to dropdown
            html.Div([
                html.Button(
                    'Scrape'
                    , id="submit"
                    , type='submit'
                    , style={
                        'color': colors['blocks'],
                        'background-color': colors['titles'],
                        'border': 'None'
                    }
                )
            ]
                , style={
                    'textAlign': 'center'
                    , 'padding': '20px'
                }
            ),
            # Loading component
            dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="circle"),
            # Maybe headlines come after this
            html.Hr(),
            html.Div(id = 'table-div')

        ]
            , className="one-half column"
            , style={
                'backgroundColor': colors['blocks']
                , 'padding': '20px'
                , 'border-radius': '5px'
                , 'box-shadow': '1px 1px 1px #9D9D9D'}
        ),
        # Right half block
        html.Div([
            # Heading of Graph area
            html.H6("Graphic Summary",
                    style={"color": colors["titles"]}),
            # Graph-1
            html.Div([
                dcc.Graph(id='graph1'
                          , style={'height': '400px'}
                          )
            ]
                , style={
                    'padding': '20px'
                }
            ),

            # Graph-2
            html.Div([
                dcc.Graph(id='graph2'
                          , style={
                        'padding': '20px'
                        , 'height': '400px'
                    }
                          )
            ])
        ]
            , className='one-half column'
            , style={
                'backgroundColor': colors["blocks"]
                , "padding": '20px'
                , 'border-radius': '5px'
                , 'box-shadow': '1px 1px 1px #9D9D9D'
            }
        )
    ]
        , className="row flex-display"
        , style={'padding': '20px'}
    ),

    # Space after content - just as an option
    html.H1(id='space2', children=' '),

    # Final paragraph
    html.Div([
        dcc.Markdown(markdown_text2
                     , style={'font-size': '12px'
                , 'color': colors["text"]
                              }
                     ),

        # Hidden div inside the app that stores the intermediate values
        html.Div(id='intermediate-value', style={'display': 'none'})
                 # Remove children after building
                 #children=df_json)
    ])

], style={'backgroundColor': colors['background']})


@app.callback(
    [Output('intermediate-value', 'children'),
    Output('loading-output-1', 'children')],
    [Input('submit', 'n_clicks')],
    [State('checklist', 'value')]
)
def scrape_and_predict(n_clicks, value):
    #Code for scrapping Hindustan Times
    if value == "HT":
        df_articles = scrap_ht()
        
    if value == "IT":
        df_articles = scrap_india_today()
        
    if value == "ET":
        df_articles = scrap_econotic_times()
    if value == "TG":
        df_articles = scrap_Guardian()
     #Code for scrapping other news websites comes here
    
    #Create features, then predict categories, then put it all together in a dataframe
    features = create_features_from_content(df_articles)
    categories = predict_from_features(features)
    df = complete_df(df_articles, categories)    
    
    return df.to_json(orient = 'split'), ' '



# Callback function to pickup data from intermediate-value and build the graphs
@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')]
)
def update_barchart(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    print(df.head())
    # Create df of category vs number of articles
    df_sum = df.groupby(["Prediction"])["title"].count()

    x = ['Politics', 'Business', 'Entertainment', 'Sports', 'Tech', 'Other']
    y = [df_sum["Politics"] if "Politics" in df_sum.index else 0,
         df_sum["Business"] if "Business" in df_sum.index else 0,
         df_sum["Entertainment"] if "Entertainment" in df_sum.index else 0,
         df_sum["Sports"] if "Sports" in df_sum.index else 0,
         df_sum["Tech"] if "Tech" in df_sum.index else 0,
         df_sum["Other"] if "Other" in df_sum.index else 0]

    figure = {
        'data': [
            {'x': x, 'y': y, 'type': 'bar', 'marker': {'color': 'rgb(62, 137, 195)'}}
        ],
        'layout' : {
            'title' : 'Number Of Articles vs Category',
            'plot_bgcolor' : colors['graph_background'],  # Remember plot is inside the paper
            'paper_bgcolor' : colors['graph_background'],
            'font' : {
                'color' : colors['text'],
                'size' : '10'
            }
        }
    }
    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')]
)
def update_piechart(jsonified_df):
    df = pd.read_json(jsonified_df, orient = 'split')

    df_sum = df.groupby(["Prediction"])["title"].count()

    x = ['Politics', 'Business', 'Entertainment', 'Sports', 'Tech', 'Other']
    y = [df_sum["Politics"] if "Politics" in df_sum.index else 0,
         df_sum["Business"] if "Business" in df_sum.index else 0,
         df_sum["Entertainment"] if "Entertainment" in df_sum.index else 0,
         df_sum["Sports"] if "Sports" in df_sum.index else 0,
         df_sum["Tech"] if "Tech" in df_sum.index else 0,
         df_sum["Other"] if "Other" in df_sum.index else 0]

    figure = {
        'data' : [
            {'values' : y,
             'labels' : x,
             #'name' : x,
             'type' : 'pie',
             'hovertemplate' : "%{label} : %{percent}<extra></extra>",
             'marker' : {
                 'colors' : ['CornflowerBlue', 'SlateBlue', 'DarkCyan', 'MidnightBlue', 'LightBlue', 'DodgerBlue']
             }
             }
        ],
        'layout' : {
            'title' : 'Category wise distribution on the cover page',
            'plot_bgcolor' : colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font' : {
                'color' : colors['text'],
                'size' : '10'}
        }
    }
    return figure

@app.callback(
    Output('table-div', 'children'),
    [Input('intermediate-value', 'children')]
)
def update_table(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    df = df[df["Prediction"] != "Other"].append(df[df["Prediction"] == "Other"],ignore_index = True)
    table_obj = dash_table.DataTable(
        id = 'table',
        columns = [{'name' : i.title(), "id" : i} for i in ["title", "Prediction"]],
        data = df.to_dict('records'),
        style_cell_conditional = [
            {'if':{'column_id' : 'title'}, 'width' : '80%'},
            {'if': {'column_id': 'Prediction'}, 'width': '20%'}
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }],
        style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
                'minWidth' : '0px', 'maxWidth' : '500px',
                'textAlign' : 'left',
                'padding' : '5px'
        },
        style_table={
            'maxHeight': '600px',
            'height' : '545px',#492 px
            #'overflowY': 'scroll',
            #'backgroundColor' : colors['blocks']
        },
        style_header = {
            'fontWeight': 'bold',
            'backgroundColor': 'rgb(230, 230, 230)'
        },
        fixed_rows={ 'headers': True, 'data': 0 }
    )

    return table_obj


if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_ui=False, dev_tools_props_check=False)

