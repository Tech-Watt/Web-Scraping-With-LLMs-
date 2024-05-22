from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
Model = 'gpt-3.5-turbo'
llm = ChatOpenAI(model=Model,api_key=api_key)

# Instantiate tools
site = 'https://www.ibm.com/topics/artificial-intelligence'
web_scrape_tool = ScrapeWebsiteTool(website_url=site)


# Create agents
web_scraper_agent = Agent(
    role='Web Scraper',
    goal='Effectively Scrape data on the websites for your company',
    backstory='''You are expert web scraper, your job is to scrape all the data for 
                your company from a given website.
                ''',
    tools=[web_scrape_tool],
    verbose=True,
    llm = llm
)


# Define tasks
web_scraper_task = Task(
    description='Scrape all the  data on the site so your company can use for decision making.',
    expected_output='All the content of the website.',
    agent=web_scraper_agent,
    output_file = 'data.txt'
)


# Assemble a crew
crew = Crew(
    agents=[web_scraper_agent],
    tasks=[web_scraper_task],
    verbose=2,
    
)

# Execute tasks
result = crew.kickoff()
print(result)

with open('results.txt', 'w') as f:
    f.write(result)