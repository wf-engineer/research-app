import streamlit as st
import openai
import google.cloud.bigquery as bigquery
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import requests
import numpy as np
from scipy.signal import savgol_filter
from datetime import datetime, date
from bs4 import BeautifulSoup
from googlesearch import search
import re

# # Install required packages
# try:
#     import openai
# except ImportError:
#     st.warning("openai package not found. Installing...")
#     !pip install --upgrade openai

if 'text_input_id' not in st.session_state:
    st.session_state['text_input_id'] = 0
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = None

checkbox_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def remove_empty_string(arr):
    return filter(lambda x: x != '', arr)

# Streamlit app
st.set_page_config(layout="wide")

st.title("Chat with IMX-GPT")
col1, col2 = st.columns([0.3, 0.7], gap="medium")

st.divider()  # 👈 Draws a horizontal rule

st.subheader("News")
news_container = st.container()

st.divider() # 👈 Draws a horizontal rule
st.subheader("Sandbox")
st.button("Sandbox", type="primary")

@st.cache_resource
def initialize():
    train_resources = requests.get("https://storage.googleapis.com/jason-key/train_resources.txt").text
    train_resources = eval(train_resources)
    openai.api_key = train_resources[1]
    # url = train_resources[2]
    # r = requests.get(url)
    # with open('credential.json', 'w') as f:
    #     f.write(r.text)
    credential_path = "credential.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    client = bigquery.Client(project='kythera-390515')
    wait = requests.get(train_resources[4]).text

    return client, train_resources, wait

client, train_resources, wait = initialize()

@st.cache_data
def chat_with_model(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=10,
    )
    return response.choices[0].text.strip()

def query(selected_date, selected_codes, selected_columns):
    if selected_columns==[]:
        return "select date_field_2, count(string_field_0) from `kythera-390515.All_claims_1.medical_claims` where (string_field_1= '{}') and date_field_2 BETWEEN '{}' AND '{}' group by date_field_2 ORDER BY date_field_2 ASC".format("' or string_field_1= '".join(selected_codes), selected_date[0], selected_date[1])
    
    base_query= "select date_field_2, count(string_field_0) from `kythera-390515.All_claims_1.medical_claims`"
    
    # Define conditions
    conditions = {

        "age": {
            "Children": "(int64_field_5 > 2002 and int64_field_5 < 2023)",
            "Adults": "(int64_field_5 > 1963 and int64_field_5 < 2002)",
            "Eldery": "(int64_field_5 < 1963)"
        },
        "gender": {
            "Male": "(string_field_7 = 'M')",
            "Female": "(string_field_7 = 'F')"
        },
        "region": {
            "Northeast": "(string_field_6 = 'Northeast')",
            "Southwest": "(string_field_6 = 'Southwest')",
            "West": "(string_field_6 = 'West')",
            "Southeast": "(string_field_6 = 'Southeast')",
            "Midwest": "(string_field_6 = 'Midwest')"
        }
    }

    all_conditions = [" or ".join([conditions[category][item] for item in selected_columns if item in conditions[category].keys()]) for category in conditions]
    
    # Append to the base query
    base_query += " where (" + ") and (".join(remove_empty_string(all_conditions)) + ")"
    base_query += " and (string_field_1= '{}') and (date_field_2 BETWEEN '{}' AND '{}') group by date_field_2 ORDER BY date_field_2 ASC"
    base_query = base_query.format("' or string_field_1= '".join(selected_codes), selected_date[0], selected_date[1])
    base_query = base_query.replace("and ()", "")

    return base_query

def query_data_and_sort(q):
    QUERY = (q)
    query_job = client.query(QUERY)
    try:
        rows = query_job.result()
    except Exception as error:
        raise error
    df = pd.DataFrame(data=[list(row.values())
                      for row in rows], columns=list(rows.schema))
    df.rename(columns={list(df.columns)[0]: 'Date', list(
        df.columns)[1]: 'Volume'}, inplace=True)
    df = df.sort_values('Date', ascending=True)
    return df

def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_text = ""

    # Extract the article content using common HTML tags
    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for paragraph in paragraphs:
        article_text += paragraph.get_text() + " "

    # Clean up the text by removing extra spaces and newlines
    article_text = re.sub(r'\s+', ' ', article_text).strip()

    # Extract the article title using title HTML tag
    titles = soup.find_all('title')
    if len(titles) == 0:
        return "", article_text
    else:
        article_title = titles[0].get_text(strip=True)
        return article_title, article_text

def google_search(keywords):
    results = []

    for keyword in keywords:
        for url in search(keyword, stop=5):  # Limiting to 5 results per keyword
            article_title, article_text = get_article_content(url)
            results.append({'title': article_title, 'link': url, 'content': article_text})

    return results

@st.cache_data()
def get_news(selected_codes):
    
    # Start the conversation
    user_input = 'Please provide a list of keywords derived form the following list of ICD10 codes: ' + ', '.join(selected_codes)
    ans = chat_with_model(user_input)
    keywords = [keyword for keyword in re.split("[\n,]", ans) if '.' in keyword or '-' in keyword][:3]
    print("KEYWORDS: ", keywords)
    search_results = google_search(keywords)

    return search_results

@st.cache_data(ttl=3600)
def process_user_input(user_input, selected_columns):
    # start conversation
    conversation = ""
    file_contents = requests.get(train_resources[5]).text
    conversation += "User: " + file_contents + "\n"
    ans = chat_with_model(conversation)
    conversation += "ChatGPT: " + ans + "\n"

    # ask openai
    conversation += "User: " + user_input + "\n"
    ans = chat_with_model(conversation)
    chain = eval(ans[ans.index('['):ans.index(']')+1])
    # get selected codes from chat-gpt ans
    selected_codes = chain[:-2]
    selected_codes = selected_codes + \
        [code.split('.')[0] for code in selected_codes if '.' in code]
    selected_codes = [code.replace(".", "") for code in selected_codes]
    selected_codes = list(set(selected_codes))
    # get selected dates from chat-gpt ans
    selected_date = chain[-2:]
    st.write("----- The following information is for debugging -----")
    st.write("Selected codes:", selected_codes)
    st.write("Selected Dates: " + selected_date[0] + ' - ' + selected_date[1])
    try:
        df = query_data_and_sort(query(selected_date, selected_codes, selected_columns))
        return df, selected_codes
    except:
        return None

def plot_data(assistant, df, alpha, option):
    try:
        shift=checkbox_options.index(option)
    except:
        shift=0
    if alpha <= 0:
        index = df
        index = index.fillna(0)
        sz = len(index)
        x = np.linspace(0.0, sz - 1, sz)
        y = index.Volume
        yhat = savgol_filter(y, 91, 3)
        plt.figure(figsize=(15, 4))
        plt.plot(x, yhat, color='red', linewidth=3)
        plt.grid(True)
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('Volume Plot')
        plt.show()
    else:
        plt.figure(figsize=(15, 4))
        original = df.set_index('Date')['Volume'].ewm(alpha=alpha).mean()
        original.plot(grid='on', label='Original')
        # Sampling every 7 values from the 'original' data frame
        sampled_original = original.iloc[shift::7]  # Select every 7th value
        # type of sampled_original
        sampled_original = sampled_original.reindex(original.index)  # Reindex with the same index as 'original'
        # Plot the sampled data frame with larger and colored dots
        sampled_original.plot(grid='on', label='Sampled ('+checkbox_options[shift]+'s)', marker='o', markersize=4, color='red')
        # Plot settings
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('Volume Plot')
        plt.show()
    assistant.pyplot(plt)

listbox = []
slider = 1.0
selected_date = [datetime(2020, 1, 1), date.today()]
selected_columns = []
checkboxes = [False,False,False,False,False,False,False,False,False,False]
df = pd.DataFrame()

def on_demographics_button_clicked():
    process_user_input(user_input, selected_date, selected_columns)

with col1:
    listbox = st.selectbox("Select days", checkbox_options)
    slider = st.slider("Select Alpha", 0.0, 1.0, 1.0, 0.05)
    sub_col1, sub_col2 = st.columns([0.4, 0.6])
    with sub_col1:
        ind = 0
        # For each column in the dataframe, create a new checkbox
        for col in ['Children', 'Adults', 'Eldery', 'Male', 'Female', 'Northeast', 'Southwest', 'West', 'Southeast', 'Midwest']:
            checkbox = st.checkbox(label=col, value=checkboxes[ind])
            checkboxes[ind] = checkbox
            if checkbox:
                selected_columns.append(col)
            ind += 1
    with sub_col2:
        demographics_button = st.button("Run Demographics", type='primary', on_click=on_demographics_button_clicked)

with col2:
    placeholder = st.empty()
    user_input = placeholder.text_input(
        label="User Input",
        placeholder="Ask IMX-GPT to create any volume index, e.g. Please create an index for covid during 2020",
        value="",
        key=st.session_state['text_input_id']
    )
    if (user_input or "") == "":
        if st.session_state['user_input'] == None:
            st.stop()
        else:
            user_input = st.session_state['user_input']
    else:
        placeholder.text_input(
            label="User Input",
            placeholder="Ask IMX-GPT to create any volume index, e.g. Please create an index for covid during 2020",
            value="",
            key=1 - st.session_state['text_input_id']
        )
        st.session_state['text_input_id'] = 1 - st.session_state['text_input_id']   
        st.session_state['user_input'] = user_input
    wait_message = random.choice(eval(wait))
    user = st.chat_message('user')
    user.write(user_input)
    assistant = st.chat_message('assistant')
    assistant.write(wait_message)
    df, selected_codes = process_user_input(user_input, selected_columns)
    download_data = df.to_csv()
    st.download_button("Download CSV", data=download_data, file_name=datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv", disabled=(download_data == ''))
    feed_button = st.button("Feed")
    if df is not None:
        plot_data(assistant, df, slider, listbox)
        search_results = get_news(selected_codes)

        if search_results:
            for result in search_results:
                if len(result['content']) > 0 and len(result['title']) > 0:
                    expander = news_container.expander(result['title'])
                    expander.write(result['content'][:300] + "...")
                    expander.write(result['link'])
        else:
            news_container.write("No results found.")
    else:
        assistant.write('Sorry! Something went wrong, try again...')

