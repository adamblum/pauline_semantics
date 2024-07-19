# compare_books_distinctverses
# find the most globally different verses in source book (as measured by cosine distance away from all other Bible verses)
# then score how similar they are to target book's most distinct verses 
#   python compare_books_distinctverses.py <source_book> <target_book> <source bible>
# <source_bible> defaults to NKJV 
#
import weaviate
import os 
import os.path
import pandas as pd 
import numpy as np
import time
import sys 
from weaviate.util import generate_uuid5 
from weaviate.classes.query import MetadataQuery,Filter 
import numpy as np
import scipy 
from scipy.spatial.distance import cosine
import time 
import timeit

def cosine_distance(v1, v2):
    #print(f"Computing distance between {v1[0:10]} and {v2[0:10]}")
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def get_verses(client,book,source_bible="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = (Filter.by_property("book").equal(book)&Filter.by_property("source").equal(source_bible)),
        limit = 10000,
        include_vector=True
    )
    #print(f"Response {response}")
    result=response.objects
    return result 

def get_all_verses(client,source_bible="NKJV"):
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = Filter.by_property("source").equal(source_bible),
        limit = 100000,
        include_vector=False
    )
    result=response.objects
    return result 

def top_10_percent_subhash(input_dict):
    # Sort the dictionary by values in descending order
    sorted_dict = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1], reverse=True)}
    # Calculate the number of elements in the top 10%
    total_values = len(sorted_dict)
    top_10_percent_threshold = total_values // 10  # 10% of total values 
    # Create a subhash with the top 10%  values
    top_10_percent_subhash = {k: v for i, (k, v) in enumerate(sorted_dict.items()) if i < top_10_percent_threshold}  
    return top_10_percent_subhash

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

def get_verse(uuid):
    # Construct the GraphQL query
    print(f"Looking for verse UUID {uuid}")
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
                book,chapter,verse,text
            }}
        }}
    }}
    """
    response = client.graphql_raw_query(query)
    get = response.get
    verse=get['Verse'][0]
    if verse: 
        print(f"Returned verse {verse}")
        return verse['book'],verse['chapter'],verse['verse'],verse["text"]
    else:
        print(f"Can't find verse {id}")
        return None

def compute_distinctness(row,num_cols,distance_matrix):
    #print(f"Computing distinctness of row {row}...")
    result = None
    tot_distance = 0 
    for col in range(num_cols):
        tot_distance += distance_matrix[row][col]
    result = tot_distance/num_cols
    #print(f"distinctness: {result}")
    return result 

# go through the hash of verse IDs with  their distinctness scores an
# find the closest match in the target book for each one
# return each of the closest match verses and the distances in an array of hashes [{"verse","closematch","distance"}]
def find_closest_matches(client,verses,source_book,source_bible="NKJV"):
    print(f"Finding closest matches in source book {source_book} for all distinctive verses from target book.")
    start=time.time()
    verse_matches=[]
    target_verses = client.collections.get("Verse")
    for verse_id in verses.keys():
        response = target_verses.query.near_object(
            near_object=verse_id,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            filters = (Filter.by_property("book").equal(source_book)&Filter.by_property("source").equal(source_bible)),
        )
        verse_match = {}
        verse_match["verse"] = verse_id
        print(f"Closematch {response.objects[0]}")
        verse_match["closematch"] = response.objects[0].properties
        verse_match["distance"] = response.objects[0].metadata.distance
        verse_matches.append(verse_match)
    end = time.time()
    elapsed = end - start
    print(f"Found closest matches in {elapsed} seconds")
    return verse_matches

def save_matches(filename,matches):
    with open(filename, "w+") as file:
        for m in matches: 
            target_book,target_chapter,target_verse,target_text=get_verse(m["verse"])
            source_book,source_chapter,source_verse,source_text=m["closematch"].properties['book'],m["closematch"].properties['chapter'],m["closematch"].properties['verse'],m["closematch"].properties["text"]
            file.write(target_book+","+target_chapter+","+target_verse+","+target_text+ \
                       ","+source_book+","+source_chapter+","+source_verse+","+source_text+","+m.metadata["distance"]+"\n")
  
# return hash of verse IDs and their vectors
def get_vectors(verses):
    print("Getting vectors for specified verses...")
    start_time = time.time()
    vectors = {}
    for v in verses:
        vectors[v.uuid] = get_vector(v.uuid)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"Elapsed time for getting vectors {elapsed_time:.2f} seconds")
    return vectors

def compute_distance_matrix(vectors):
    print("Computing distance matrix")
    start = time.time()
    ids = list(vectors.keys())
    num_vectors = len(ids)
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((num_vectors, num_vectors))
    
    # Compute cosine distance between each pair of vectors
    for i in range(num_vectors):
        for j in range(i, num_vectors):  # compute only upper triangle
            vec_i = vectors[ids[i]]
            vec_j = vectors[ids[j]]
            if i == j:
                distance = 0.0  # distance to self is 0
            else:
                distance = cosine(vec_i, vec_j)
            
            # Fill in the symmetric distance matrix
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    end = time.time()
    elapsed_time = end - start
    print("Computed distance matrix in {elapsed_time} seconds")
    return distance_matrix

def verse_number(verse,verses):
    print(f"Finding {verse.uuid} in verses")
    num = 0
    for v in verses:
        if v.uuid == verse.uuid:
            return num
        num += 1
    num = -1 # not found
    return num

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']
source_bible="NKJV"
results_dir = "./analysis/"
distance_matrix_file = results_dir + "distance_matrix.npy"

client = weaviate.connect_to_wcs(
        cluster_url = WEAVIATE_SERVER,  
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY"))
    )

if len(sys.argv)>=2:
    source_book = sys.argv[1]
    print(f"Finding close match verses in source book: {source_book}")
else:
    print(f"Source book is required")
    exit()

if len(sys.argv)>=3:
    target_book = sys.argv[2]
    print(f"For most distinctive verses in target book: {target_book}")
else:
    print(f"Target book is required")
    exit()

if len(sys.argv)>=4:
    source_bible = sys.argv[3]

try: 
    start_time = time.time()
    all_verses = get_all_verses(client,source_bible)
    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"Elapsed time for getting all {len(all_verses)} verses: {elapsed_time:.2f} seconds")
    if os.path.isfile(distance_matrix_file):
        start_time = time.time()
        print(f"Loading existing distance matrix... ")
        distance_matrix = np.load(distance_matrix_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if distance_matrix.shape!=(len(all_verses),len(all_verses)):
            print(f"Stored distance matrix is wrong dimensionality: {distance_matrix.shape}")
            exit()  
        print(f"Loaded existing matrix with dimensionality {distance_matrix.shape}. Elapsed time {elapsed_time:.2f} seconds")
    else: 
        start_time = time.time()
        all_vectors = get_vectors(all_verses)
        _,distance_matrix = compute_distance_matrix(all_vectors)
        np.save(distance_matrix_file,distance_matrix)
        end_time = time.time()
        elapsed_time = end_time - start_time  
        print(f"Elapsed time for getting vectors, computing distance matrix and saving it: {elapsed_time:.2f} seconds")        


    # use this to find the most distinct verses in the specified TARGET book
    target_verses = get_verses(client,target_book,source_bible)
    print(f"Target book verses count {len(target_verses)}")
    distinctnesses={}
    num_verses = len(all_verses)
    start_time = time.time()
    verse_num = verse_number(target_verses[0],all_verses)
    if verse_num < 0:
        print(f"Can't find verse {target_verses[0]}. Stopping.")
        exit()
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Found verse number: {verse_num}. Elapsed time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    print(f"Finding most distinct verses in target book {target_book}")
    row = num_verses * verse_num
    for row in range(num_verses): 
        v = all_verses[row]
        distinctness = compute_distinctness(row,num_verses,distance_matrix)
        distinctnesses[v.uuid] = distinctness
        row += 1 
    most_distinct = top_10_percent_subhash(distinctnesses)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Found most distinct verses in target book {target_book}. Elapsed time: {elapsed_time:.2f} seconds")
    #print(f"Most distinctive verses {most_distinct}")

    # now find the closest matches for each of the distinct verses in the target book in the source book
    closest_matches=find_closest_matches(client,most_distinct,source_book,source_bible)
    results_file = results_dir + "distinct_verse_from_target_" + target_book + "-close_matches_in_source_" + source_book + ".csv"
    # save them for further analysis: to determine likelihood target was written by author of source
    save_matches(results_file,closest_matches)
finally:
    client.close()


