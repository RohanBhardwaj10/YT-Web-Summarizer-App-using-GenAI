import validators
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.schema import Document



st.set_page_config(page_title="Summarize YT/Web ~ tool to summarize text from Youtube & Web",page_icon="📖")
st.title("📖 Summarize YT/Web ~ tool to summarize text from Youtube & Web")
st.subheader("Summarize URL")

import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key=st.secrets["GROQ_API_KEY"]
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt_gen='''
Provide a summary of about 300 to 350 words on the following content and give a title for the summary too:
Context :{text}
'''
def extract_video_id(url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return video_id_match.group(1) if video_id_match else None


prompt=PromptTemplate(input_variables=['text'],template=prompt_gen)

txt_url=st.text_input("Enter the URL for Video or Website")

if st.button("Let's GO!"):
    if not  txt_url.strip():
        st.error("Please enter the required details.")
    elif not validators.url(txt_url):
        st.error("Enter the correct and valid URL.")
    else:
        with st.spinner("Loading..."):
                if "youtube.com" in txt_url or "youtu.be" in txt_url:
                    video_id = extract_video_id(txt_url)
                    if not video_id:
                        st.error("Could not extract video ID.")
                    else:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        text = " ".join([t['text'] for t in transcript])
                        docs = [Document(page_content=text)]
                else:
                    loader=UnstructuredURLLoader(urls=[txt_url],ssl_verify=False,
                                                 headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})
                    docs=loader.load()
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                summary=chain.run(docs)
                st.success(summary)

        

