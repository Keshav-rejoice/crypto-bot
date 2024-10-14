
from llama_agents import (AgentService,AgentOrchestrator,ControlPlaneServer,LocalLauncher,SimpleMessageQueue,HumanService,QueueMessage,CallableMessageConsumer,ServerLauncher)
from llama_index.core.agent import  FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.agent import ReActAgent
import yfinance as yf
import matplotlib.pyplot as plt
import os
import tiktoken

import re
import pandas as pd
import streamlit as st
import openai
import asyncio
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from llama_index.core.schema import TransformComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext,Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import openai

os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]


pc = Pinecone(api_key="264040b3-b298-4918-9d56-b31134d5ba48")
index = pc.Index("crypto-bot")
print(index)
st.title("Crypto Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_the_secret_fact()->str:
  """Return the Secret Fact Key"""
  return "The secret fact is :A baby llama is called a 'cria"

def get_crypto_price(symbol):
    try:
        data = yf.Ticker(f"{symbol}-USD").history(period='1y')
        if data.empty:
            return f"No data available for {symbol}"
        return str(data.iloc[-1].Close)
    except Exception as e:
        return f"Error fetching crypto price for {symbol}: {str(e)}"

def get_crypto_volume(symbol):
    try:
        data = yf.Ticker(f"{symbol}-USD").history(period='1y')
        if data.empty:
            return f"No data available for {symbol}"
        return str(data.iloc[-1].Volume)
    except Exception as e:
        return f"Error fetching crypto volume for {symbol}: {str(e)}"

def get_market_cap(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        market_cap = ticker.info['marketCap']
        return f"{market_cap:,}"
    except Exception as e:
        return f"Error fetching market cap for {symbol}: {str(e)}"

def get_circulating_supply(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        circulating_supply = ticker.info['circulatingSupply']
        return f"{circulating_supply:,}"
    except Exception as e:
        return f"Error fetching circulating supply for {symbol}: {str(e)}"
    
def calculate_SMA(symbol, window):
    data = yf.Ticker(f"{symbol}-USD").history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(symbol, window):
    data = yf.Ticker(f"{symbol}-USD").history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(symbol):
    data = yf.Ticker(f"{symbol}-USD").history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])

def calculate_MACD(symbol):
    data = yf.Ticker(f"{symbol}-USD").history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]},{signal[-1]},{MACD_histogram[-1]}'

def plot_crypto_price(symbol):
    try:
        data = yf.Ticker(f"{symbol}-USD").history(period='1y')
        if data.empty:
            return f"No data available for {symbol}"
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data.Close)
        plt.title(f'{symbol} price over last year')
        plt.xlabel("Date")
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.savefig('crypto_price.png')
        plt.close()
        return "Chart created successfully"
    except Exception as e:
        return f"Error plotting crypto price for {symbol}: {str(e)}"
    
def get_24h_change(symbol):
    try:
        data = yf.Ticker(f"{symbol}-USD").history(period='1y')
        if data.empty or len(data) < 2:
            return f"Insufficient data available for {symbol}"
        yesterday_close = data.iloc[-2].Close
        today_close = data.iloc[-1].Close
        change = ((today_close - yesterday_close) / yesterday_close) * 100
        return f"{change:.2f}%"
    except Exception as e:
        return f"Error calculating 24h change for {symbol}: {str(e)}"

def get_all_time_high(symbol):
    try:
        data = yf.Ticker(f"{symbol}-USD").history(period='max')
        if data.empty:
            return f"No data available for {symbol}"
        ath = data.Close.max()
        ath_date = data.Close.idxmax().strftime('%Y-%m-%d')
        return f"${ath:.2f} on {ath_date}"
    except Exception as e:
        return f"Error fetching all-time high for {symbol}: {str(e)}"
    
def generate_market_report(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        data = ticker.history(period='1d')
        info = ticker.info

        report = {
            'Symbol': symbol,
            'Price (USD)': data.iloc[-1].Close,
            'Market Cap': info.get('marketCap', 'N/A'),
            'Circulating Supply': info.get('circulatingSupply', 'N/A'),
            '24h Volume': data.iloc[-1].Volume,
            '24h Change (%)': get_24h_change(symbol),
            'RSI': calculate_RSI(symbol),
            '50-day SMA': calculate_SMA(symbol, 50),
            '200-day SMA': calculate_SMA(symbol, 200),
            'All-Time High': get_all_time_high(symbol)
        }

        return pd.DataFrame([report]).T.reset_index()
    except Exception as e:
        return f"Error generating market report for {symbol}: {str(e)}"
# CRYPTO_ANALYST_PERSONA = """
# You are an expert cryptocurrency trader with 8 years of experience. Your knowledge consists of:
# - Deep understanding of blockchain technology and its applications
# - Extensive knowledge of various cryptocurrencies, their use cases, and market dynamics
# - Strong analytical skills for technical and fundamental analysis
# - Experience with trading strategies, risk management, and market psychology
# - Up-to-date information on regulatory developments and market trends
# Provide insights, analysis, and recommendations based on your expertise. Use relevant terminology and explain your reasoning clearly.
# """

# custom_query_prompt = PromptTemplate(
#     """
#     {crypto_analyst_persona}

#     Context information is below.
#     ---------------------
#     {context_str}
#     ---------------------

#     Given the context information and not prior knowledge, answer the query.
#     Query: {query_str}

#     Your response should include:
#     1. A concise overview of the topic
#     2. Key factors or concepts to consider
#     3. Relevant market data or trends
#     4. Potential implications or future outlook
#     5. Any caveats or limitations to the analysis

#     Answer:
#     """
# )

# custom_refine_prompt = PromptTemplate(
#     """
#     {crypto_analyst_persona}

#     We have provided an existing answer: {existing_answer}

#     We have the opportunity to refine the existing answer with some more context below.
#     ------------
#     {context_msg}
#     ------------

#     Given the new context, refine the original answer to better address the query: {query_str}. 
#     If the context isn't useful, return the original answer.

#     Refined Answer:
#     """
# )
documents = SimpleDirectoryReader("data").load_data()

# Define TextCleaner class
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r'[^A-Za-z0-9 ]+', '', node.text.lower())
        return nodes
tokenizer = tiktoken.get_encoding("cl100k_base")   
def split_text_by_tokens(text, max_tokens=6000):
    """Splits text into smaller chunks based on token count."""
    tokens = tokenizer.encode(text)  # Tokenize the input text
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)  # Convert tokens back to text
        yield chunk_text

text_cleaner = TextCleaner()
chunked_documents = []
for document in documents:
    for chunk in split_text_by_tokens(document.text, max_tokens=8192): 
        chunked_documents.append(Document(text=chunk))  

vector_store = PineconeVectorStore(
    pinecone_index=index,
)
Settings.chunk_size = 512
settings = Settings.embed_model = OpenAIEmbedding()


# Create index
text_cleaner = TextCleaner()
index = VectorStoreIndex.from_documents(chunked_documents, vector_store=vector_store, settings=settings, transformations=[text_cleaner])
chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="keshav",
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=index.as_retriever(similarity_top_k=10),
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    # text_qa_template=custom_query_prompt,
    # refine_template=custom_refine_prompt,
    llm=OpenAI(model="gpt-4", max_tokens=600, temperature=0.0),
    memory=chat_memory
)

# Initialize agent
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="CryptoCoin_Expert_Analysis",
        description="Useful for understanding blockchain and analyzing cryptocurrency market risks."
    ),
)

tool = FunctionTool.from_defaults(fn=get_the_secret_fact)
cryptoprice_tools = FunctionTool.from_defaults(fn=get_crypto_price)
cryptovolume_tools = FunctionTool.from_defaults(fn = get_crypto_volume)
marketcap_tools = FunctionTool.from_defaults(fn = get_market_cap)
circulatingsupply_tools = FunctionTool.from_defaults(fn=get_circulating_supply)
calculatesma_tools = FunctionTool.from_defaults(fn=calculate_SMA)
calculateema_tools = FunctionTool.from_defaults(fn =calculate_EMA)
calculatersi_tools = FunctionTool.from_defaults(fn=calculate_RSI)
calculatemacd_tools = FunctionTool.from_defaults(fn=calculate_MACD)
plot_crypto_price_tools = FunctionTool.from_defaults(fn=plot_crypto_price)
get_24h_change_tools = FunctionTool.from_defaults(fn=get_24h_change)
get_all_time_high_tools = FunctionTool.from_defaults(fn=get_all_time_high)
generate_market_report_tools =  FunctionTool.from_defaults(fn=generate_market_report)
llm = OpenAI(model="gpt-4o-2024-08-06", temperature=0)

agent = ReActAgent.from_tools([query_engine_tool,cryptoprice_tools,cryptovolume_tools,marketcap_tools,generate_market_report_tools,circulatingsupply_tools,calculatesma_tools,calculatersi_tools,calculatemacd_tools,plot_crypto_price_tools,get_24h_change_tools,get_all_time_high_tools], llm=llm, verbose=True)




if prompt := st.chat_input("Ask about cryptocoins?"):
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = agent.chat(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

