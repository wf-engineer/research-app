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

# # Install required packages
# try:
#     import openai
# except ImportError:
#     st.warning("openai package not found. Installing...")
#     !pip install --upgrade openai

if 'text_input_id' not in st.session_state:
    st.session_state['text_input_id'] = 0
    st.session_state['user_input'] = None

checkbox_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Streamlit app
st.set_page_config(layout="wide")

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

def query(ans, structure):
    chain = eval(ans[ans.index('['):ans.index(']')+1])
    selected_codes = chain[:-2]
    selected_date = chain[-2:]
    selected_codes = selected_codes + \
        [code.split('.')[0] for code in selected_codes if '.' in code]
    selected_codes = [code.replace(".", "") for code in selected_codes]
    selected_codes = list(set(selected_codes))

    st.write("These are the specific codes being used for this index:", selected_codes)

    return structure.format("' or string_field_1= '".join(selected_codes), selected_date[0], selected_date[1])

def query_data_and_sort(q):
    QUERY = (q)
    query_job = client.query(QUERY)
    rows = query_job.result()
    df = pd.DataFrame(data=[list(row.values())
                      for row in rows], columns=list(rows.schema))
    df.rename(columns={list(df.columns)[0]: 'Date', list(
        df.columns)[1]: 'Volume'}, inplace=True)
    df = df.sort_values('Date', ascending=True)
    return df

@st.cache_data(ttl=3600)
def process_user_input(user_input):
    # start conversation
    conversation = ""
    file_contents = requests.get(train_resources[5]).text
    conversation += "User: " + file_contents + "\n"
    ans = chat_with_model(conversation)
    conversation += "ChatGPT: " + ans + "\n"

    # ask openai
    conversation += "User: " + user_input + "\n"
    ans = chat_with_model(conversation)
    try:
        df = query_data_and_sort(query(ans, train_resources[0]))
        return df
    except Exception as e:
        print(e)
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
        print("TYPE OF SAMPLED ORIGINAL", type(sampled_original))
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

def download_data(df, alpha, options):
    try:
        shift=list(options)[0]
    except:
        shift=0
    if alpha <= 0:
        index = df.fillna(0)
        sz = len(index)
        x = np.linspace(0.0, sz - 1, sz)
        y = index.Volume
        yhat = savgol_filter(y, 91, 3)
        result = pd.DataFrame({'Date': df['Date'], 'Volume': yhat})
    else:
        result = df.set_index('Date')['Volume'].ewm(alpha=alpha).mean().reset_index()
        result2 = result.iloc[shift::7]   

    result.to_csv('dowloads/index.csv', index=False)
    result2.to_csv('dowloads/index_sampled.csv', index=False)
    print('Downloaded')
 

col1, col2 = st.columns([0.3, 0.7], gap="medium")

listbox = []
slider = 1.0
df = pd.DataFrame()

def on_download_button_clicked():
    download_data(df, slider, listbox)

with col1:
    listbox = st.selectbox("Select days", checkbox_options)
    slider = st.slider("Select Alpha", 0.0, 1.0, 1.0, 0.05)
    butt_col1, butt_col2 = st.columns(2)
    with butt_col1:
        download_button = st.button("Download", on_click=on_download_button_clicked)
    with butt_col2:
        feed_button = st.button("Feed")

with col2:
    placeholder = st.empty()
    user_input = placeholder.text_input(
        label="User Input",
        placeholder="Ask IMX-GPT to create any volume index, e.g. Please create an index for covid during 2020",
        value="",
        key=st.session_state['text_input_id']
    )
    print("USER INPUT: ", user_input)
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
    df = process_user_input(user_input)
    if df is not None:
        plot_data(assistant, df, slider, listbox)
    else:
        assistant.write('Sorry! Something went wrong, try again...')
