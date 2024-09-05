import streamlit as st
import feedparser
import pandas as pd
import urllib.parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Define the Google News Feed Scraper class
class GoogleNewsFeedScraper:
    def __init__(self, query):
        self.query = query
        self.data = []

    def scrape_google_news_feed(self):
        encoded_query = urllib.parse.quote(self.query)
        rss_url = f'https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en'
        feed = feedparser.parse(rss_url)

        if feed.entries:
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                description = entry.description
                pubdate = entry.published
                source = entry.source.title if 'source' in entry else 'Unknown'

                # Append the data to the list
                self.data.append({
                    'Title': title,
                    'Link': link,
                    'Description': description,
                    'Published': pubdate,
                    'Source': source
                })
        else:
            st.write("Nothing Found!")

    def to_dataframe(self):
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(self.data)
        
        # Convert the 'Published' column to datetime format
        df['Published'] = pd.to_datetime(df['Published'])
        
        # Get the current date and calculate the date one week ago
        one_week_ago = datetime.now() - timedelta(days=7)
        
        # Filter the DataFrame to include only rows from the last week
        df = df[df['Published'] >= one_week_ago]
        
        # Sort the DataFrame by the 'Published' column in descending order
        df = df.sort_values(by='Published', ascending=False)
        
        return df

# Streamlit app starts here
st.title('Google News Sentiment Analysis')

# User input for commodity selection
commodity = st.selectbox('Select a commodity:', ['Copper', 'Lithium'])

# Determine the query and corresponding CSV file based on the selected commodity
if commodity == 'Copper':
    query = 'copper price'
    forecast_file = 'copper_forecast.csv'
    commodity_name = 'Copper'
else:
    query = 'lithium price'
    forecast_file = 'lithium_forecast.csv'
    commodity_name = 'Lithium'

# Scrape the news data
scraper = GoogleNewsFeedScraper(query)
scraper.scrape_google_news_feed()
df = scraper.to_dataframe()

# Display the sentiment analysis
if not df.empty:
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Lists to store sentiment scores
    negative = []
    neutral = []
    positive = []

    # Analyze the sentiment of the title and description
    for n in range(df.shape[0]):
        title = df.iloc[n, 0]
        description = df.iloc[n, 2]  # Access the 'Description' column
        title_analyzed = analyzer.polarity_scores(title)
        description_analyzed = analyzer.polarity_scores(description)
        
        # Calculate the average sentiment scores for title and description
        negative.append((title_analyzed['neg'] + description_analyzed['neg']) / 2)
        neutral.append((title_analyzed['neu'] + description_analyzed['neu']) / 2)
        positive.append((title_analyzed['pos'] + description_analyzed['pos']) / 2)

    # Add the sentiment scores to the DataFrame
    df['Negative'] = negative
    df['Neutral'] = neutral
    df['Positive'] = positive

    # Calculate the sentiment statistics
    negative_mean = df['Negative'].mean() * 100
    neutral_mean = df['Neutral'].mean() * 100
    positive_mean = df['Positive'].mean() * 100

    # Display sentiment statistics with colored text
    st.subheader('Sentiment Analysis Summary')
    st.markdown(f"**Negative:** <span style='color:red;'>{negative_mean:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Neutral:** {neutral_mean:.2f}%", unsafe_allow_html=True)
    st.markdown(f"**Positive:** <span style='color:green;'>{positive_mean:.2f}%</span>", unsafe_allow_html=True)

    # Sort the DataFrame by the 'Positive' sentiment score in descending order
    df_sorted_positive = df.sort_values(by='Positive', ascending=False)
    df_sorted_negative = df.sort_values(by='Negative', ascending=False)

    # Select the top 5 positive and top 5 negative articles
    top_5_positive_articles = df_sorted_positive.head(5)
    top_5_negative_articles = df_sorted_negative.head(5)

    # Display expandable section for top 5 positive articles
    with st.expander("Top 5 Positive Articles"):
        for index, row in top_5_positive_articles.iterrows():
            st.write(f"**Title:** {row['Title']}")
            st.write(f"**Link:** {row['Link']}")
            st.write(f"**Published:** {row['Published']}")
            st.write("---")

    # Display expandable section for top 5 negative articles
    with st.expander("Top 5 Negative Articles"):
        for index, row in top_5_negative_articles.iterrows():
            st.write(f"**Title:** {row['Title']}")
            st.write(f"**Link:** {row['Link']}")
            st.write(f"**Published:** {row['Published']}")
            st.write("---")

    # Display the data
    with st.expander("All Articles"):
        st.subheader('News Articles')
        st.dataframe(df)
else:
    st.write("No articles found for this query.")

# Load the saved CSV file with comparison data
comparison_df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)

# Dynamically filter the past year of data and the upcoming 3 months
end_date = comparison_df.index.max()
start_date = end_date - pd.DateOffset(years=1)
forecast_end_date = end_date + pd.DateOffset(months=3)
comparison_df = comparison_df.loc[start_date:end_date]

# Apply smoothing with rolling mean
smoothed_actual = comparison_df['Actual'].rolling(window=1).mean()
smoothed_future_predict = comparison_df['Future Predict'].rolling(window=1).mean()

# Create plotly traces for each series
trace_actual = go.Scatter(x=smoothed_actual.index, y=smoothed_actual, mode='lines', name='Actual Close Price', line=dict(color='white'))
trace_future_predict = go.Scatter(x=smoothed_future_predict.index, y=smoothed_future_predict, mode='lines', name='Future Predict', line=dict(color='green'))

# Create the layout with a black background and no grid
layout = go.Layout(
    title=f'Actual vs Predicted {commodity_name} Prices (Past Year and Upcoming 3 Months)',
    xaxis_title='Date',
    yaxis_title=f'{commodity_name} Price',
    legend=dict(x=0, y=1.0),
    margin=dict(l=40, r=0, t=40, b=30),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),  # Change the font color to white for visibility on black background
    xaxis=dict(showgrid=False),  # Disable gridlines on x-axis
    yaxis=dict(showgrid=False)   # Disable gridlines on y-axis
)

# Create the figure
fig = go.Figure(data=[trace_actual, trace_future_predict], layout=layout)


# Display the plot in Streamlit
st.plotly_chart(fig)


# Load the copper inventory data
df_inventory = pd.read_csv('copper_inventory.csv', index_col=0, parse_dates=True)

# Plot the inventory data
fig_inventory = go.Figure()

# Add traces for production and usage with specified colors
fig_inventory.add_trace(go.Scatter(x=df_inventory.index, y=df_inventory['Production'], mode='lines', name='Production', line=dict(color='green')))
fig_inventory.add_trace(go.Scatter(x=df_inventory.index, y=df_inventory['Usage'], mode='lines', name='Usage', line=dict(color='red')))

# Update layout for better visualization
fig_inventory.update_layout(
    title='Copper Production and Usage Over the Years',
    xaxis_title='Year',
    yaxis_title='Quantity',
    legend=dict(x=0, y=1.0),
    margin=dict(l=40, r=0, t=40, b=30),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# Display the plot in Streamlit
st.plotly_chart(fig_inventory)
