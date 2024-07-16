# compare_books
# Generate comparison scores of each verse in source book with closely related verses in target book
#
#   python compare_books.py <source_book_number> <target_book_number> <source>
#
import weaviate
import os 
import pandas as pd 
import numpy as np
import time
import sys 
from weaviate.util import generate_uuid5 
from weaviate.classes.query import MetadataQuery,Filter 

# find verses from target book that are similar to specified verse
def find_similar(client,uuid,target_book,source_book=None,chapter=None,verse=None,source="NKJV"):
    coll = client.collections.get("Verse")
    if uuid == None: 
        obj_uuid = generate_uuid5({"book":source_book,"chapter":chapter,"verse":verse})
    else:
        obj_uuid = uuid 
    print(f"UUID: {obj_uuid}")
    verse = coll.query.fetch_object_by_id(uuid=obj_uuid)
    print(f"Retrieved object {verse}")

    # now go search target_book for close matches to meaning of this verse
    response = coll.query.near_object(
                near_object=obj_uuid,
                limit=5,
                distance=0.12,
                return_metadata=MetadataQuery(distance=True),
                filters=(Filter.by_property("book").equal(target)&Filter.by_property("source").equal(source))
        )
    print(f"\n\nSearch target: {verse.properties}\n------------")
    print(f"\nResults:\n------------")
    for o in response.objects:
        print(f"Distance {o.metadata.distance}, Verse {o.properties}")
    return response.objects    

def get_verses(client,book,source="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = (Filter.by_property("book").equal(book)&Filter.by_property("source").equal(source)),
        limit = 10000 
    )
    result=response.objects
    return result 

def save_similarity_results(filename,source_verse,target_verses):
    with open(filename, "a+") as file:
        for v in target_verses: 
            file.write(source_verse.properties['book']+","+source_verse.properties['chapter']+","+source_verse.properties['verse']+","+v.properties['book']+","+v.properties['chapter']+","+v.properties['verse']+","+str(v.metadata.distance)+"\n")
  

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']

client = weaviate.connect_to_wcs(
        cluster_url = WEAVIATE_SERVER,  
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY"))
    )

source_book = sys.argv[1]
if len(sys.argv)<3:
    target_book = 43
else:
    target_book = sys.argv[2]

source="NKJV"
results_dir = "../analysis/"
if len(sys.argv)>=4:
    source=sys.argv[3]

try: 
    source_verses = get_verses(client,source_book,source)
    results_file = results_dir + source_book + '-' + target_book + '.csv'
    for v in source_verses: 
        similar_verses = find_similar(client,v.uuid,target_book,source)
        if similar_verses:
            save_similarity_results(results_file,v,similar_verses)
        else:
            print(f'No similar verses for {v}')
finally:
    client.close()


