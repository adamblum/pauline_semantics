import numpy as np
import os,openai
import weaviate
from weaviate.util import generate_uuid5 
from weaviate.classes.query import MetadataQuery,Filter 

openai.api_key = os.getenv("OPENAI_APIKEY")

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']

# List of books in the correct order
books_order = [
    'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
    'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
    '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job',
    'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah',
    'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos',
    'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
    'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
    '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians', 'Philippians',
    'Colossians', '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy',
    'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John',
    '2 John', '3 John', 'Jude', 'Revelation'
]

# Function to get the index of a book in the books_order list
def get_book_index(book):
    return books_order.index(book)

def get_verses(client,book,source_bible="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = (Filter.by_property("book").equal(book)&Filter.by_property("source").equal(source_bible)),
        limit = 10000,
        include_vector=True
    )
    #print(f"Response {response}")
    sorted_verses = sorted(response.objects, key=lambda x: (get_book_index(x.properties["book"]), int(x.properties["chapter"]), int(x.properties["verse"])))
    return sorted_verses

def get_all_pauline_verses(client):
    verses = get_verses(client,"Romans")
    print(f"Num verses {len(verses)}")
    return verses

def get_embedding_openai(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.embeddings.create(input = [text], model=model).data[0].embedding

def calculate_raw_complexity(embedding):
    # Calculate the variance of the embedding vector
    variance = np.var(embedding)
    return variance

client = weaviate.connect_to_wcs(
        cluster_url = WEAVIATE_SERVER,  
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY"))
    )

# List of verses from Pauline epistles (for example purposes, few verses are provided)
pauline_epistle_verses = get_all_pauline_verses(client)
raw_complexity_scores = []

for verse in pauline_epistle_verses:
    #print(f"Verse {verse}")
    embedding = get_embedding_openai(verse.properties['text'])
    raw_complexity = calculate_raw_complexity(embedding)
    raw_complexity_scores.append(raw_complexity)

# Calculate mean and standard deviation of raw complexity scores
mean_raw_complexity = np.mean(raw_complexity_scores)
std_dev_raw_complexity = np.std(raw_complexity_scores)

print(f"Mean raw complexity score: {mean_raw_complexity}")
print(f"Standard deviation of raw complexity scores: {std_dev_raw_complexity}")