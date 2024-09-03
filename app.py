import streamlit as st
from textblob import TextBlob
import pandas as pd 
import emoji
from bs4 import BeautifulSoup
from urllib.request import urlopen
from wordcloud import WordCloud
import matplotlib.pyplot as plt


@st.cache_data
def get_text(raw_url):
    try:
        page = urlopen(raw_url)
        soup = BeautifulSoup(page, "html.parser")
        fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
        return fetched_text
    except Exception as e:
        st.error("An error occurred while fetching the URL.")
        return None


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud


def main():
    """Sentiment Analysis Emoji App"""

    st.title("Sentiment Analysis Emoji App")

    activities = ["Sentiment", "Text Analysis on URL", "About"]
    choice = st.sidebar.selectbox("Choice", activities)

    if choice == 'Sentiment':
        st.subheader("Sentiment Analysis")
        st.write(emoji.emojize('Everyone :red_heart: Streamlit', use_aliases=True))
        raw_text = st.text_area("Enter Your Text", "Type Here")
        
        if st.button("Analyze"):
            blob = TextBlob(raw_text)
            result = blob.sentiment.polarity
            
            if result > 0.0:
                custom_emoji = ':smile:'
            elif result < 0.0:
                custom_emoji = ':disappointed:'
            else:
                custom_emoji = ':expressionless:'
                
            st.write(emoji.emojize(custom_emoji, use_aliases=True))
            st.info(f"Polarity Score is: {result}")
            
            # Display word cloud
            wordcloud = generate_wordcloud(raw_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot(plt)

    if choice == 'Text Analysis on URL':
        st.subheader("Analysis on Text From URL")
        raw_url = st.text_input("Enter URL Here", "Type here")
        text_preview_length = st.slider("Length to Preview", 50, 100)
        
        if st.button("Analyze"):
            if raw_url != "Type here":
                result = get_text(raw_url)
                
                if result:
                    blob = TextBlob(result)
                    len_of_full_text = len(result)
                    len_of_short_text = round(len(result) / text_preview_length)
                    
                    st.success(f"Length of Full Text: {len_of_full_text} characters")
                    st.success(f"Length of Short Text: {len_of_short_text} characters")
                    st.info(result[:len_of_short_text])
                    
                    # Convert Sentence objects to strings
                    c_sentences = [str(sent) for sent in blob.sentences]
                    c_sentiment = [sent.sentiment.polarity for sent in blob.sentences]
                    
                    new_df = pd.DataFrame(zip(c_sentences, c_sentiment), columns=['Sentence', 'Sentiment'])
                    st.dataframe(new_df)
                    
                    # Sentiment distribution visualization
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(new_df['Sentiment'])
                    
                    # Display basic text statistics
                    st.subheader("Text Statistics")
                    st.write(f"Number of words: {len(result.split())}")
                    st.write(f"Number of sentences: {len(blob.sentences)}")
                    
                    # Display word cloud
                    st.subheader("Word Cloud")
                    wordcloud = generate_wordcloud(result)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis('off')
                    st.pyplot(plt)

    if choice == 'About':
        st.text("Created by Shailendra Singh")
        st.text("Contact: shailendrasingh1703@gmail.com")


if __name__ == '__main__':
    main()
