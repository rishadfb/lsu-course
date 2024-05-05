import os
import pathlib

import pandas
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

DOMAIN = "developer.mozilla.org"


def remove_newlines(series: pandas.Series):
    series = series.str.replace('\n', ' ')
    series = series.str.replace('\\n', ' ')
    series = series.str.replace('  ', ' ')
    series = series.str.replace('  ', ' ')
    return series


project_root = pathlib.Path(__file__).parent.parent.resolve()
TEXT_DIRECTORY = project_root / 'text' / DOMAIN
PROCESSED_DIRECTORY = project_root / 'processed'
SCRAPED_CSV_FILE_PATH = PROCESSED_DIRECTORY / 'scraped.csv'
EMBEDDINGS_CSV_FILE_PATH = PROCESSED_DIRECTORY / 'embeddings.csv'

# Create a list to store the text files
texts = []

# Get all the text files in the text directory
for file in os.listdir(TEXT_DIRECTORY):

    file_to_open = TEXT_DIRECTORY / file
    print(file_to_open)

    # Open the file and read the text
    with open(file_to_open, "r", encoding="UTF-8") as f:
        text = f.read()
        # we replace the last 4 characters to get rid of .txt, and replace _ with / to generate the URLs we scraped
        filename = file[:-4].replace('_', '/')

        """
        There are a lot of contributor.txt files that got included in the scrape, this weeds them out. There are also a lot of auth required urls that have been scraped to weed out as well
        """
        if filename.endswith(".txt") or 'users/fxa/login' in filename:
            continue

        # then we replace underscores with / to get the actual links so we can cite contributions
        texts.append(
            (filename, text))

# Create a dataframe from the list of texts
data_frame = pandas.DataFrame(texts, columns=['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
data_frame['text'] = data_frame.fname + ". " + remove_newlines(data_frame.text)

data_frame.to_csv(SCRAPED_CSV_FILE_PATH)

tokenizer = tiktoken.get_encoding("cl100k_base")

data_frame = pandas.read_csv(SCRAPED_CSV_FILE_PATH, index_col=0)
data_frame.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
data_frame['n_tokens'] = data_frame.text.apply(
    lambda x: len(tokenizer.encode(x)))

chunk_size = 1000  # Max number of tokens

text_splitter = RecursiveCharacterTextSplitter(
    # This could be replaced with a token counting function if needed
    length_function=len,
    chunk_size=chunk_size,
    chunk_overlap=0,  # No overlap between chunks
    add_start_index=False,  # We don't need start index in this case
)

shortened = []

for row in data_frame.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > chunk_size:
        # Split the text using LangChain's text splitter
        chunks = text_splitter.create_documents([row[1]['text']])
        # Append the content of each chunk to the 'shortened' list
        for chunk in chunks:
            shortened.append(chunk.page_content)

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

data_frame = pandas.DataFrame(shortened, columns=['text'])
data_frame['n_tokens'] = data_frame.text.apply(
    lambda x: len(tokenizer.encode(x)))


data_frame['embeddings'] = data_frame.text.apply(lambda x: openai.embeddings.create(
    input=x, model='text-embedding-ada-002').data[0].embedding)

data_frame.to_csv(EMBEDDINGS_CSV_FILE_PATH)