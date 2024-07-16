import weaviate
import os 
import pandas as pd 
import numpy as np
import time
import sys 
from weaviate.util import generate_uuid5 
from weaviate.classes.query import MetadataQuery,Filter 
import numpy as np

def cosine_distance(v1, v2):
    print(f"Computing distance between {v1[0:10]} and {v2[0:10]}")
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def get_verses(client,book,source="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = Filter.by_property("book").equal(book),
        limit = 10000,
        include_vector=True
    )
    #print(f"Response {response}")
    result=response.objects
    return result 

def get_all_verses(client,source="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = Filter.by_property("source").equal(source),
        limit = 100000,
        include_vector=False
    )
    result=response.objects
    return result 

def top_1_percent_subhash(input_dict):
    # Sort the dictionary by values in descending order
    sorted_dict = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Calculate the number of elements in the top 1%
    total_values = len(sorted_dict)
    top_1_percent_threshold = total_values // 100  # 1% of total values
    
    # Create a subhash with the top 1% values
    top_1_percent_subhash = {k: v for i, (k, v) in enumerate(sorted_dict.items()) if i < top_1_percent_threshold}
    
    return top_1_percent_subhash
# for given verse compute the average cosine of this verse against ALL OTHER VERSES
# in Old and New Testaments for that verse

def get_vector(uuid):
    # Construct the GraphQL query
    query = f"""
    {{
        Get {{
            {"Verse"}(
            where: {{
                path: ["id"]
                operator: Equal
                valueString: "{uuid}"
            }}
            ) {{
                _additional {{
                    vector
                }}
            }}
        }}
    }}
    """
    response = client.graphql_raw_query(query)
    get = response.get
    #print(f"Result {result}")
    vector = get['Verse'][0]['_additional']['vector']
    #print(f"Returned vector {vector[0:10]}")
    return vector 

def compute_uniqueness(verse,all_verses):
    result = None
    tot_distance = 0 
    for target in all_verses:
        target_vector = get_vector(target.uuid)
        tot_distance+=cosine_distance(verse.vector['default'],target_vector)
    result = tot_distance/len(all_verses)
    print(f"Uniqueness: {result}")
    return result 

# go through the list of verses and find the closest match in the target book for each one
# return each of the closest match verses and the distances in a hash {"verse","closematch","distance"}
def find_closest_matches(client,verses,target):
    verse_matches={}
    target_verses = client.collections.get("Verse")
    for v in verses:
        response = target_verses.query.near_object(
            near_object=v.uuid,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            filters = (Filter.by_property("book").equal(target)&Filter.by_property("source").equal(source)),
        )
        verse_match = {}
        verse_match["verse"] = v
        verse_match["closematch"] = response.objects[0].properties
        verse_match["distance"] = response.objects[0].distance
        verse_matches.append(verse_match)
    return verse_matches

def save_matches(filename,matches):
    with open(filename, "w+") as file:
        for m in matches: 
            file.write(m["verse"]+","+m["closematch"]+","+m["distance"]+"\n")
  

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']

client = weaviate.connect_to_wcs(
        cluster_url = WEAVIATE_SERVER,  
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY"))
    )

if len(sys.argv)>=2:
    source_book = sys.argv[1]
    print(f"Analyzing verses from {source_book}")
else:
    print(f"Source book is required")
    exit()

if len(sys.argv)>=3:
    target = sys.argv[2]
else:
    print(f"Target book is required")
    exit()

source="NKJV"
results_dir = "../../analysis"
if len(sys.argv)>=4:
    source = sys.argv[3]

try: 
    source_verses = get_verses(client,source_book,source)
    print(f"Source book verses count {len(source_verses)}")
    all_verses = get_all_verses(client,source)
    print(f"All verses count {len(all_verses)}")
    uniquenesses={}
    for v in source_verses: 
        uniqueness = compute_uniqueness(v,all_verses)
        uniquenesses[v.uuid] = uniqueness
    most_unique = top_1_percent_subhash(uniquenesses)
    print(f"Most unique verses {most_unique}")
    closest_matches=find_closest_matches(client,most_unique,target)
    results_file = results_dir + "uniques_from" + source + "_in_" + target + ".csv"
    save_matches(results_file,closest_matches)
finally:
    client.close()


