import csv
import json
import math

# ==========================================
# STEP 1: LOAD ALIASES
# ==========================================
with open('skill_aliases.json', 'r', encoding='utf-8') as f:
    skill_aliases = json.load(f)

# ==========================================
# STEP 2: LOAD & NORMALIZE RESUMES
# ==========================================
normalized_resumes = []

with open('resumes.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_skills_string = row['Raw Skills']
        # Split by comma, strip spaces, lowercase
        tokens = [token.strip().lower() for token in raw_skills_string.split(',')]
        
        normalized_skills = set()
        for token in tokens:
            if token in skill_aliases:
                normalized_skills.add(skill_aliases[token])
                
        normalized_resumes.append({
            'id': row['ID'],
            'name': row['Candidate'],
            'skills': list(normalized_skills)
        })

# ==========================================
# STEP 3: VOCABULARY CONSTRUCTION
# ==========================================
shared_vocabulary = set()
for resume in normalized_resumes:
    for skill in resume['skills']:
        shared_vocabulary.add(skill)

sorted_vocab = sorted(list(shared_vocabulary))

# ==========================================
# STEP 4: TF-IDF VECTOR CONSTRUCTION (Resumes Only)
# ==========================================
df_counts = {skill: 0 for skill in sorted_vocab}
for resume in normalized_resumes:
    for skill in resume['skills']:
        df_counts[skill] += 1

idf_scores = {}
for skill in sorted_vocab:
    idf_scores[skill] = math.log(10 / df_counts[skill])

for resume in normalized_resumes:
    N = len(resume['skills'])
    tf = 1 / N if N > 0 else 0
    
    vector = []
    for skill in sorted_vocab:
        if skill in resume['skills']:
            vector.append(tf * idf_scores[skill])
        else:
            vector.append(0.0)
            
    resume['vector'] = vector

# ==========================================
# STEP 5: BUILD JD VECTORS (Binary)
# ==========================================
jds = []

with open('job_descriptions.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Combine required and preferred skills, then split by comma
        raw_jd_skills = row['Required Skills'] + "," + row['Preferred Skills']
        tokens = [token.strip().lower() for token in raw_jd_skills.split(',')]
        
        normalized_jd_skills = set()
        for token in tokens:
            if token in skill_aliases:
                normalized_jd_skills.add(skill_aliases[token])
        
        binary_vector = []
        for skill in sorted_vocab:
            if skill in normalized_jd_skills:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
                
        jds.append({
            'jd_id': row['JD'],
            'company': row['Company'],
            'role': row['Role'],
            'vector': binary_vector
        })

# ==========================================
# STEP 6: COSINE SIMILARITY & RANKING
# ==========================================
def calculate_magnitude(vector):
    return math.sqrt(sum(val ** 2 for val in vector))

def calculate_dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

print("\n" + "="*40)
print("FINAL HACKATHON RESULTS")
print("="*40)

for jd in jds:
    results = []
    jd_vector = jd['vector']
    jd_magnitude = calculate_magnitude(jd_vector)
    
    for resume in normalized_resumes:
        resume_vector = resume['vector']
        resume_magnitude = calculate_magnitude(resume_vector)
        
        if jd_magnitude == 0 or resume_magnitude == 0:
            similarity = 0.0
        else:
            dot_product = calculate_dot_product(resume_vector, jd_vector)
            similarity = dot_product / (resume_magnitude * jd_magnitude)
            
        results.append({
            'name': resume['name'],
            'score': round(similarity, 2)
        })
    
    # Sort first by score (Highest to Lowest), then alphabetically by Name (A-Z)
    results.sort(key=lambda x: (-x['score'], x['name']))
    
    top_3 = results[:3]
    
    print(f"\n{jd['jd_id']}")
    print(f"{jd['company']} ({jd['role']})")
    
    formatted_output = ", ".join([f"{candidate['name']} ({candidate['score']:.2f})" for candidate in top_3])
    print(formatted_output)