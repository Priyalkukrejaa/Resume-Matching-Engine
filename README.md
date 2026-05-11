# Resume-Matching-Engine
## Overview
This project is a custom Resume Matching Engine built for the Redrob AI Campus Hackathon. It evaluates 10 noisy resume datasets from Indian university students and matches them against 3 Job Descriptions (JDs) from Korean technology companies. 

The entire engine is built using **pure Python standard libraries** (`csv`, `json`, `math`). No external libraries like `pandas`, `numpy`, or `scikit-learn` were used, adhering strictly to the competition rules.

## Project Structure
* `main.py`: The core engine that processes the data, calculates vectors, and outputs the rankings.
* `resumes.csv`: The dataset containing 10 candidate resumes with raw, noisy skills.
* `job_descriptions.csv`: The dataset containing 3 JDs with required and preferred skills.
* `skill_aliases.json`: The standard dictionary used to map noisy tokens to canonical skills.

## How It Works (The Pipeline)

### 1. Skill Normalization & Deduplication
The engine reads the raw skills from the resumes and JDs. It converts all text to lowercase, strips whitespace, and maps tokens to their canonical forms using the provided JSON dictionary. Crucially, it processes multi-word phrases (like "machine learning") before single tokens. Unknown tokens are discarded, and the final list for each candidate is deduplicated.

### 2. Shared Vocabulary Construction
A single, shared vocabulary is generated from the normalized skills of all 10 candidates. This vocabulary is sorted alphabetically to ensure perfect alignment for the mathematical vectors.

### 3. TF-IDF Vectorization
The engine treats the 10 resumes as the complete corpus and generates TF-IDF vectors for them.
* **Term Frequency (TF):** Calculated as $TF = 1 / N$, where $N$ is the total unique skills in a specific resume.
* **Inverse Document Frequency (IDF):** Calculated as $IDF = \ln(10 / df)$, using the natural logarithm without smoothing.

### 4. Binary JD Vectors & Cosine Similarity
Job descriptions are converted into binary vectors (1 if the skill is present, 0 if not). The engine then calculates the cosine similarity between each candidate's TF-IDF vector and the JD's binary vector. 

### 5. Final Ranking & Tie-Breakers
Candidates are ranked based on their similarity score, rounded to exactly 2 decimal places. In the event of a mathematical tie, the engine automatically breaks the tie alphabetically by the candidate's name to determine the Top 3.

## How to Run
Ensure all four files are in the same directory. Open your terminal, navigate to the folder, and run:

```bash
python main.py
