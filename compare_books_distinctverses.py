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
import csv 

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
    sorted_verses = sorted(response.objects, key=lambda x: (get_book_index(x.properties["book"]), int(x.properties["chapter"]), int(x.properties["verse"])))
    #print(f"First verse {sorted_verses[0]}")
    return sorted_verses

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
    '2 John', '3 John', 'Jude', 'Revelation','3 Corinthians','Laodiceans'
]

# Function to get the index of a book in the books_order list
def get_book_index(book):
    return books_order.index(book)

def get_all_verses(client,bible="NKJV"):
    print(f"Getting all verses from {bible}")
    coll = client.collections.get("Verse")
    response = coll.query.fetch_objects(
        filters = Filter.by_property("source").equal(bible),
        limit = 100000,
        include_vector=False
    )
    sorted_verses = sorted(response.objects, key=lambda x: (get_book_index(x.properties["book"]), int(x.properties["chapter"]), int(x.properties["verse"])))
    print(f"Number of all verses: {len(sorted_verses)}")
    return sorted_verses

def top_20_percent_subhash(input_dict):
    # Sort the dictionary by values in descending order
    sorted_dict = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1], reverse=True)}
    # Calculate the number of elements in the top 20 percent
    total_values = len(sorted_dict)
    top_20_percent_threshold = total_values // 5  # 10% of total values 
    # Create a subhash with the top 20%  values
    top_20_percent_subhash = {k: v for i, (k, v) in enumerate(sorted_dict.items()) if i < top_20_percent_threshold}  
    return top_20_percent_subhash

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
    #print(f"Looking for verse UUID {uuid}")
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
        #print(f"Returned verse {verse}")
        return verse['book'],verse['chapter'],verse['verse'],verse["text"]
    else:
        #print(f"Can't find verse {id}")
        return None

def compute_distance(col,first_row,num_rows,matrix):
    #print(f"Computing distance of source row {row}...")
    result = None
    tot_distance = 0 
    for row in range(num_rows):
        #print(f"Add distance for {row},{col}")
        tot_distance += matrix[row+first_row][col]
    result = tot_distance/num_rows
    #print(f"distance: {result}")
    return result 

def isopenorclose(i,numverses):
    if (i==0) or (i==1):
        return True
    if (i==numverses-1) or (i==numverses-2):
        return True
    return False

# go through the hash of verse IDs with  their distinctness scores an
# find the closest match in the source book for each one
# return each of the closest match verses and the distances in an array of hashes [{"verse","closematch","distance"}]
def find_closest_matches(client,verses,source_book,source_bible="NKJV"):
    print(f"Finding closest matches in source book {source_book} for all distinctive verses from target book.")
    start=time.time()
    verse_matches=[]
    source_verses = client.collections.get("Verse")
    num = 0
    for verse_id in verses.keys():
        response = source_verses.query.near_object(
            near_object=verse_id,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
            filters = (Filter.by_property("book").equal(source_book)&Filter.by_property("source").equal(source_bible)),
        )
        verse_match = {}
        verse_match["verse"] = verse_id
        #print(f"Closematch {response.objects[0]}")
        verse_match["closematch"] = response.objects[0].properties
        verse_match["distance"] = response.objects[0].metadata.distance
        verse_matches.append(verse_match)
    end = time.time()
    elapsed = end - start
    print(f"Found closest matches in {elapsed} seconds")
    return verse_matches

def save_matches(filename,matches,most_distinct,complexities):
    with open(filename, "w+") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['TargetBook','Chapter','Verse','Text','SourceBook','Chapter','Verse','Text','Distance','Distinctness','Complexity'])
        i = 0
        tot_distance = 0
        for m in matches: 
            target_book,target_chapter,target_verse,target_text=get_verse(m["verse"])
            source_book,source_chapter,source_verse,source_text=m["closematch"]['book'],m["closematch"]['chapter'],m["closematch"]['verse'],m["closematch"]["text"]
            writer.writerow([target_book,target_chapter,target_verse,target_text,source_book,source_chapter,source_verse,source_text,str(m["distance"]),most_distinct[m["verse"]],complexities[m["verse"]]])
            tot_distance += m["distance"]
            i += 1
        avg_distance = tot_distance/len(matches)
        file.write("AVERAGE:,,,,,,,"+str(avg_distance))
  
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
    print(f"Computed distance matrix in {elapsed_time} seconds")
    return distance_matrix

def compute_hetero_matrix(source_vectors,target_vectors):
    print("Computing heterogeneous (source v. target bible) distance matrix")
    start = time.time()
    source_ids = list(source_vectors.keys())
    num_source_vectors = len(source_ids)
    target_ids = list(target_vectors.keys())
    num_target_vectors = len(target_ids)

    distance_matrix = np.zeros((num_source_vectors, num_target_vectors))
    # Compute cosine distance between each pair of vectors
    for i in range(num_source_vectors):
        vec_i = source_vectors[source_ids[i]]
        for j in range(num_target_vectors): 
            vec_j = target_vectors[target_ids[j]] 
            distance = cosine(vec_i, vec_j)  
            distance_matrix[i, j] = distance
    end = time.time()
    elapsed_time = end - start
    print(f"Computed distance matrix in {elapsed_time} seconds")
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

# given a hash of verse IDs plus the distinct scores, saved to a CSV with book,chapter,verse,score
def save_distinct_scores(csv_file_name,data,complexities):
    # Open the CSV file for writing
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Verse ID','Book','Chapter','Verse','Text','Score','Complexity'])
        
        # Write the data
        for verse_id, score in data.items():
            book,chapter,verse,text = get_verse(verse_id)
            complexity=complexities[verse_id]
            writer.writerow([verse_id,book,chapter,verse,text,score,complexity])

def verse_string(verse_object):
    result = verse_object.properties['book'] + ' ' + verse_object.properties['chapter'] + ' ' + verse_object.properties['verse']
    return result

def normalize_and_scale_complexity(raw_complexity, mean=0.0006505854642018747, std_dev=3.248088534511439e-08, target_mean=5, target_std_dev=2, min_score=0, max_score=10):
    # Standardize the raw complexity to have a mean of `mean` and std deviation of `std_dev`
    standardized_complexity = (raw_complexity - mean) / std_dev
    # Scale to the desired range with a target mean and std deviation
    scaled_complexity = standardized_complexity * target_std_dev + target_mean
    # Ensure the value falls within the [min_score, max_score] range
    final_score = np.clip(scaled_complexity, min_score, max_score)
    return final_score

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']
results_dir = "./analysis/"
distinct_scores_file = results_dir + "distinct_scores.csv"
WEIGHT_COMPLEXITY=0.005

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

source_bible="NKJV"
if len(sys.argv)>=4:
    source_bible = sys.argv[3]

target_bible = source_bible
if len(sys.argv)>=5:
    target_bible = sys.argv[4]

distance_matrix_file = results_dir + source_bible + "-" + target_bible + "_distance_matrix.npy"

try: 
    start_time = time.time()
    all_source_verses = get_all_verses(client,source_bible)
    if source_bible != target_bible:
        all_target_verses = get_all_verses(client,target_bible)
    else:
        all_target_verses = all_source_verses
    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"Elapsed time for getting all verses: {elapsed_time:.2f} seconds")

    if os.path.isfile(distance_matrix_file):
        start_time = time.time()
        print(f"Loading existing distance matrix: {distance_matrix_file}... ")
        distance_matrix = np.load(distance_matrix_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if distance_matrix.shape!=(len(all_source_verses),len(all_target_verses)):
            print(f"Stored distance matrix is wrong dimensionality: {distance_matrix.shape}")
            exit()  
        print(f"Loaded existing matrix with dimensionality {distance_matrix.shape}. Elapsed time {elapsed_time:.2f} seconds")
    else: 
        start_time = time.time()
        source_vectors = get_vectors(all_source_verses)
        if source_bible == target_bible:
            distance_matrix = compute_distance_matrix(source_vectors)
        else: 
            target_vectors = get_vectors(all_target_verses)
            distance_matrix = compute_hetero_matrix(source_vectors,target_vectors)

        np.save(distance_matrix_file,distance_matrix)
        end_time = time.time()
        elapsed_time = end_time - start_time  
        print(f"Elapsed time for getting vectors, computing distance matrix and saving it: {elapsed_time:.2f} seconds")  

    # use this to find the most distinct verses in the specified TARGET book
    distinctnesses={}
    complexities={}
    start_time = time.time()
    source_verses = get_verses(client,source_book,source_bible)
    num_source_verses = len(source_verses)
    target_verses = get_verses(client,target_book,target_bible)
    num_target_verses = len(target_verses)
    source_verse_num = verse_number(source_verses[0],all_source_verses)
    if source_verse_num < 0:
        print(f"Can't find source verse {source_verses[0]}. Stopping.")
        exit()
    target_verse_num = verse_number(target_verses[0],all_target_verses)
    if target_verse_num < 0:
        print(f"Can't find target verse {target_verses[0]}. Stopping.")
        exit()
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Found verse number: {target_verse_num}. Elapsed time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    print(f"Finding most distinct verses in target book: {target_book}")
    for i in range(num_target_verses):
        if isopenorclose(i,num_target_verses):
            #print(f"Skipping {i}")
            continue
        v = target_verses[i]
        #print(f"{i}-th source verse: {verse_string(v)}") 
        # compute distinctness as distance + weight_complexity * complexity, 
        # where complexity measures the variance of the vecctor
        vector = get_vector(v.uuid)
        if target_verse_num+i >= len(all_target_verses):
            print(f"Target verse {target_verse_num+i} greater than number of target verses {len(all_target_verses)}")
            break 
        distinctness = compute_distance(target_verse_num+i,source_verse_num,num_source_verses,distance_matrix) 
        distinctness += WEIGHT_COMPLEXITY * normalize_and_scale_complexity(np.var(vector))
        distinctnesses[v.uuid] = distinctness
        complexities[v.uuid] = normalize_and_scale_complexity(np.var(vector))

    most_distinct = top_20_percent_subhash(distinctnesses)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Found most distinct verses in target book {target_book}. Elapsed time: {elapsed_time:.2f} seconds")
    save_distinct_scores(distinct_scores_file,most_distinct,complexities)

    # now find the closest matches for each of the distinct verses in the target book in the source book
    closest_matches=find_closest_matches(client,most_distinct,source_book,source_bible)
    results_file = results_dir + "distinct_verse_from_target_" + target_book + "-close_matches_in_source_" + source_book + ".csv"
    # save them for further analysis: to determine likelihood target was written by author of source
    save_matches(results_file,closest_matches,most_distinct,complexities)
finally:
    client.close()


