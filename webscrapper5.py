import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool
import streamlit as st
from crewai_tools import tool
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt


# Initialize the Groq LLM (replace with your actual API key)
os.environ["GROQ_API_KEY"] = "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"
llm = 'groq/llama3-70b-8192'

# Define folder paths
long_para_folder = './hacker-news'
tweet_folder = './Tweeter'

# Create the folders if they don't exist
os.makedirs(long_para_folder, exist_ok=True)
os.makedirs(tweet_folder, exist_ok=True)

# Function to read file content manually
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def fetch_and_scrape_hacker_news_stories(num_stories: int = 15) -> List[Dict[str, Optional[str]]]:
    """
    Fetches the top stories from Hacker News and scrapes their content.
    """
    response = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()[:num_stories]

    stories = []
    for story_id in story_ids:
        story_response = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
        story_data = story_response.json()
        if 'title' in story_data and 'url' in story_data:
            try:
                content_response = requests.get(story_data['url'], headers={'User-Agent': 'MyCustomUserAgent/1.0'})
                content_response.raise_for_status()
                soup = BeautifulSoup(content_response.text, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
            except Exception as e:
                print(f"Error scraping {story_data['url']}: {e}")
                content = None
            stories.append({'title': story_data['title'], 'content': content})
    
    return stories

# Fetch and scrape stories from Hacker News
stories = fetch_and_scrape_hacker_news_stories()

# Agent for writing long paragraphs
writer_long_para = Agent(
    role='Content Writer',
    goal='3 to 4 paragraphs from Hacker News stories.',
    backstory='A skilled writer with a talent for interesting and engaging content.',
    tools=[DirectoryReadTool(directory=long_para_folder), FileReadTool()],
    verbose=True,
    llm=llm
)

# Task for writing long paragraphs
write_long_para = Task(
    description='''
    Write the title, url and 3 to 4 paragraph summary of the stories provided in the content.
    Ensure that:
    - The summary is focused purely on the content of the story without unnecessary sections like 'Introduction' or 'Conclusion'.
    - Avoid mentioning filenames, file paths in the summary.
    - The content is targetted towards a technical profesional audience, give the output from third person perspective.
    - Each story should be summarized in a direct, concise, and natural flow without unnecessary formatting.
    - Only the important points of the story should be highlighted.
    - Use clear, informative language without adding any placeholders or prompts like "briefly mention".
    ''',
    expected_output='A markdown file with title and url with a well-structured 3-4 paragraph summary, free from irrelevant sections or placeholders.',
    agent=writer_long_para,
    output_file='hacker-news/long-para.md'  
)


# Agent for creating tweets
writer_tweeter = Agent(
    role='Twitter Content Curator',
    goal='Most Interesting Tweet with 280 characters in Tweeter Folder 1 txt file for each tweet',
    backstory='A skilled content curator with a knack for creating concise, engaging tweets.',
    tools=[DirectoryReadTool(directory=long_para_folder), DirectoryReadTool(directory=tweet_folder), FileReadTool()],
    verbose=True,
    llm=llm
)

# Task for tweet generation

write_tweeter = Task(
    description='''
    Read hacker-news/long-para.md file, select top 10 interesting AI-related paragraphs.
    Create a summary in the Twitter folder, with only 1 .txt file named "Summary Tweets".
    Summaries should be interesting and engaging.
    Generate the output in markdown.
    Summaries should:
    - Be concise (around 280 characters).
    - Use engaging language.
    - Highlight key AI-related points.
    - The summaries should be tech featured.
    - Include relevant hashtags (e.g., #AI, #Tech).
    - Avoid redundancy, ensuring each tweet covers different information.
    - Optionally include a call-to-action like “Read more.”
    ''',
    expected_output='A short summary formatted in markdown.',
    agent=writer_tweeter,
    output_file=f'{tweet_folder}/Summary_Tweets.txt'  # Single output file
)

# Create a combined crew for writing long paragraphs and tweets
crew_combined = Crew(
    agents=[writer_long_para, writer_tweeter],  # Both agents in one crew
    tasks=[write_long_para, write_tweeter],     # Both tasks handled by the same crew
    verbose=True,
    manager_llm=llm
)

# Fetch and scrape stories from Hacker News
stories = fetch_and_scrape_hacker_news_stories()

# Kickoff the process for both writing long paragraphs and generating tweets
crew_combined.kickoff(inputs={'stories': stories})

# Read the generated tweet summary
with open(f'{tweet_folder}/Summary_Tweets.txt', 'r') as tweet_file:
    tweets = tweet_file.read()

# Streamlit to display the tweets
st.title("Generated AI Tweets")
st.write(tweets)