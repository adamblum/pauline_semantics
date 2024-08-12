# verse_counts
#
# For given source return the number of verses for each book.  
#
# Usage: 
#   python verse_counts.py <source bible>
#
# Example:
#   python verse_count.py NKJV 
import openai
import os, sys 

import weaviate
import weaviate.classes as wvc

from weaviate.util import generate_uuid5 
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']
MIN_LENGTH = 10

def list_counts(counts):
  for k in counts.keys():
    print(f"Book {k}: {counts[k]}")

source = "NKJV"
if (len(sys.argv)>1):
  source = sys.argv[1]
  print(f"Source is {source}")

client = weaviate.connect_to_wcs(
        cluster_url = WEAVIATE_SERVER,  
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY"))
    

try:
  collection = client.collections.get("Verse")

  response = collection.query.fetch_objects(
    filters=(
      Filter.by_property("source").equal(source) 
    ),
    limit=100000
  )
  i = 0
  verse_counts = {}
  blank_counts = {}
  blanks=[]
  for o in response.objects: 
    #print(f"Object: {o}")
    props = o.properties
    if props["book"] not in verse_counts.keys():
      verse_counts[props["book"]] = 1
    else:
      verse_counts[props["book"]] += 1 
    if props["text"] is None or len(str(props["text"]))<MIN_LENGTH:
      blanks.append(props)
      if props["book"] not in blank_counts.keys():
        blank_counts[props["book"]] = 1
      else:
        blank_counts[props["book"]] += 1 
    i += 1 
finally:
    client.close()

print("\nVerse counts:\n==========================")
list_counts(verse_counts)
print("\nBlank counts\n===========================")
list_counts(blank_counts)
print(f"Blank verses:\n===========================")
for b in blanks:
  print(f"Blank verse {b}")
print(f"{i} verses have been analyzed.")