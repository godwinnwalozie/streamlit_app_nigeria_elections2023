from ast import increment_lineno
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import os
import matplotlib
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import collections
from nltk.corpus import stopwords



st.set_page_config(layout= "wide")
st.subheader("𝐒𝐞𝐧𝐭𝐢𝐦𝐞𝐧𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 (𝐍𝐋𝐏) 𝐨𝐟 𝐏𝐫𝐞𝐬𝐢𝐝𝐞𝐧𝐭𝐢𝐚𝐥 𝐂𝐚𝐧𝐝𝐢𝐝𝐚𝐭𝐞𝐬 - 𝟐𝟎𝟐𝟑 𝐍𝐢𝐠𝐞𝐫𝐢𝐚 🇳🇬  𝐄𝐥𝐞𝐜𝐭𝐢𝐨𝐧" )


# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 3rem;
                    padding-bottom: 0.3rem;
                }
                .css-hxt7ib {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
   
        </style>
        """, unsafe_allow_html=True)


# button styling
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: cornflowerblue;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #6F84FF;
    color:#ffffff
    }
</style>""", unsafe_allow_html=True)




# import dataset
path = os.path.abspath(os.path.dirname(__file__))
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_dataset():
    data =pd.read_csv((os.path.join(path,"influencers.csv")))
    return data
dataset = load_dataset()
tweet_number = len(dataset)
obi_mention = dataset[dataset.name.str.contains("Obi")].value_counts().sum()
atiku_mention = dataset[dataset.name.str.contains("Atiku")].value_counts().sum()
tinubu_mention = dataset[dataset.name.str.contains("Tinubu")].value_counts().sum()

col1, col2, col3,col4 = st.columns(4)
col1.metric(label="Total number of tweets extracted", value=tweet_number, delta="")
col2.metric(label="PeterObi mentions", value= obi_mention, delta="")
col3.metric(label="Atiku mentions", value= atiku_mention, delta="")
col4.metric(label="Tinubu mentions", value= tinubu_mention, delta="")

st.success("###### ❝𝐊𝐞𝐲 𝐰𝐨𝐫𝐝 𝐬𝐞𝐚𝐫𝐜𝐡 𝐢𝐧𝐜𝐥𝐮𝐝𝐞𝐬 : @𝐏𝐞𝐭𝐞𝐫𝐎𝐛𝐢,𝐨𝐛𝐞𝐝𝐢𝐞𝐧𝐭, 𝐁𝐀𝐓, 𝐓𝐢𝐧𝐮𝐛𝐮,@𝐨𝐟𝐟𝐢𝐜𝐢𝐚𝐥𝐁𝐀𝐓,𝐀𝐬𝐢𝐰𝐚𝐣𝐮, @𝐚𝐭𝐢𝐤𝐮, 𝐣𝐚𝐠𝐚𝐛𝐚𝐧, 𝐢𝐧𝐞𝐜𝐧𝐢𝐠𝐞𝐫𝐢𝐚❞")

def photo_atiku ():
    dir_name = os.path.abspath(os.path.dirname(__file__))
    file = Image.open(os.path.join(dir_name,"atiku.jpg"))
    im1 = st.sidebar.image(file)
    return im1


def photo_obi ():
    dir_name = os.path.abspath(os.path.dirname(__file__))
    file = Image.open(os.path.join(dir_name,"peterobi.jpg"))
    im2 = st.sidebar.image(file)
    return im2


def photo_bat ():
    dir_name = os.path.abspath(os.path.dirname(__file__))
    file = Image.open(os.path.join(dir_name,"bat1.jpg"))
    im3 = st.sidebar.image(file)
    return im3

with st.sidebar:
    st.subheader("Metrics")

    option  = st.sidebar.radio('Select a candidate',('Peter Obi (LP)', 'Bola Ahmed Tinubu (APC)', 'Atiku Abubakar (PDP)'))

    if  option == "Atiku Abubakar (PDP)":
        name = "Atiku"
        im1 = photo_atiku()
    elif option == 'Peter Obi (LP)':
        name = 'Obi'
        im2= photo_obi()
    else :
        name = 'Tinubu' 
        im3 = photo_bat()

dataset1 = dataset[dataset.name == name]


with st.container():
    col1, col2, = st.columns(2)
    with col1:
             
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            
        def each1 ():
            sent_tab = dataset1.loc[:,["name","sentiment"]]
            fig=px.pie(sent_tab.sentiment.value_counts(), names = sent_tab.sentiment.value_counts().index, values \
                = sent_tab.sentiment.value_counts().values,hole=.3)
            fig.update_traces(textposition='outside', textinfo='percent+label',textfont_size=15,pull=0.09)
            fig.update_layout(width = 475,height = 480, annotations=[dict(text= name, x=0.50, y=0.5, font_size=20, showarrow=False,\
                )],legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1),title = '<b>Candidate Sentiment</b>')
            return fig
        plot1 = each1()
        st.write(plot1)
        
        dir_name = os.path.abspath(os.path.dirname(__file__))
        mask = np.array(Image.open(os.path.join(dir_name,"mask1.png")))
        
    
    
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def wcl():
            stopwords = STOPWORDS
            stopwords.update(["https", "co","I","The","s","u","go","us","obi","Tinubu","atiku","peter","will","nigeria"])
            plt.subplots (figsize = (16,8))
            text = "".join(dataset1.loc[:,["tweet_preprocessed","sentiment"]][dataset1.loc[:,\
                ["tweet_preprocessed","sentiment"]].sentiment == 'positive'].tweet_preprocessed)
            fig, ax = plt.subplots(figsize = (20,20))
            wc = WordCloud(mask = mask,stopwords=STOPWORDS, max_words = 1000 , collocations=False ,\
                max_font_size=100, scale=10,relative_scaling=.6, background_color="black", \
                    colormap = "rainbow",random_state=42).generate(text)
            plt.axis("off")
            plt.tight_layout(pad=0)
            ax.imshow(wc,interpolation="bilinear")
            plt.title("What people are saying about aspirant", fontsize = 40, fontweight ="bold")
            plt.show()
            return fig
        plot3 = wcl()
        st.write(plot3)
        
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def loc():
            tweet_loc = dataset.loc[:,['location',"sentiment"]].location.value_counts().head(10).sort_values().to_frame()
            fig=px.bar(tweet_loc, barmode='group',orientation='v',color_discrete_sequence=px.colors.qualitative.G10)
            fig.update_layout(  title = '<b>Top Locations</b>',title_x=0.1,\
                legend=dict(orientation="v", yanchor="bottom",y=1.02,xanchor="right",x=1),showlegend=False)
            fig.update_traces(textposition='outside',textfont_size=11)
            return fig
        plot5 = loc()
        st.write(plot5)
    
        
        
    with col2:  
        
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def all ():
            sent_tab = pd.crosstab(dataset.name, dataset.sentiment).transpose()
            fig = px.bar(sent_tab,barmode='group',orientation='v',text_auto=True)
            fig.update_layout( width = 670, title = '<b>How They Compare</b>',title_x=0.1,\
                legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
            fig.update_traces(textfont_size=11,textposition='inside')
            return fig
        plot2 = all()
        st.write(plot2)

        
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def media():
            tweet_platform = dataset1['Source of Tweet'].value_counts().head(4).sort_values().to_frame()
            fig =px.bar(tweet_platform, color= ["green","blue","orange","black"],text_auto=True, orientation='v',\
                color_discrete_sequence=px.colors.qualitative.G10)
            fig.update_layout( title = '<b>What decices are people tweeting with ?</b>',title_x=0.05,\
                legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1),showlegend=False)
            fig.update_traces(textposition='inside',textfont_size=11)
            return fig
        plot4 = media()
        st.write(plot4)
        

        
        

           


        
                
       


        






    
    
    
    
    
    # with col2 :
    #     @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    #     def sentiment_all2():
    #         sent_tab = pd.crosstab(dataset.name, dataset.sentiment).transpose()
    #         fig = px.bar(sent_tab,barmode='group',text_auto=True)
    #         fig.update_layout( width = 650,title = 'Social Media Sentiments',title_x=0.5)
    #         fig.update_traces(textposition='outside',textfont_size=11)
    #         return fig
    #     plot2 = sentiment_all2()
    #     st.write(plot2)
        
    # sent_tab = pd.crosstab(dataset1.name, dataset1.sentiment).transpose()
 
       
           
