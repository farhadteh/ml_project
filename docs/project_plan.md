Part 1: Project Brief
This section outlines the high-level plan for our search engine prototype.

Goal 🎯
The primary goal is to build a fast, relevant, and intelligent search engine prototype. It should demonstrate modern search relevance techniques and be able to rank creative assets (like templates, photos, or graphics) based on a user's text query.

Project Scope
We will build a self-contained, in-memory search engine within a single Python file. The solution will rely only on Python's standard library, making it easy to run anywhere without installation. It will operate on a small, representative mock dataset of about 10-15 documents. The entire development is designed to be completed within a 1-hour interview timeframe.

Core Features ✨
The engine will implement a multi-stage ranking pipeline to ensure high-quality results:

Keyword Relevance (BM25): The foundation of our ranking. This standard algorithm ranks documents based on how well keywords in the query match the content in document fields like the title and tags. Important fields will be given a higher weight.

Synonym Expansion: To improve recall, the engine will understand that a search for "cv" is also a search for "resume." It will expand queries with related terms, giving a slight preference to the original term.

Facet-Based Boosting: To better understand user intent, the engine will recognize special keywords (facets) like size ("A4"), style ("minimal"), or color ("gold") and apply a significant score boost to matching items.

Popularity Signal: A small score bonus will be added for items with high popularity, ensuring that proven, well-liked assets get a slight edge.

Result Diversity (MMR): To improve the user experience, the final results will be re-ranked using Maximal Marginal Relevance (MMR). This ensures the top results are not just near-duplicates of each other, providing a more varied and useful selection.

Constraints 🚧
To align with the project's scope and goals, we will adhere to the following constraints:

Pure Functional Approach: The entire solution will be built using functions. No classes will be used, which makes the code simple, testable, and demonstrates clear data flow.



Single File: The complete, runnable solution will be contained in one .py file.

Part 2: AI-Assisted Development Plan
This is a step-by-step guide to building the project using an AI coding assistant (like Cursor or Copilot). Each step includes its goal and a specific, high-leverage prompt to give the AI.

Step 1: Scaffolding and Configuration
Goal: Create the basic file structure, define the mock data, and set up a central configuration dictionary. This gets the boilerplate code out of the way instantly.

AI Prompt:

"Generate a single Python file named search_engine.py. Inside, create a CONFIG dictionary to hold all search parameters. Also, create a DOCS list containing 10-15 mock documents representing Canva assets, each with an id, title, tags, desc, popularity, size, and style."

Step 2: Preprocessing and Indexing
Goal: Create a single, pure function that does all the heavy lifting upfront. This function will build our "search context," which includes the inverted index for fast keyword lookups and other pre-computed data. This is the core of our functional approach.

AI Prompt:

"Write a pure Python function create_search_context(docs, config). This function should take the list of documents and the config dictionary. It must return a new dictionary containing all pre-computed data needed for searching, including:

An inverted index for BM25.

Calculated document lengths and average document lengths per field.

TF-IDF vectors for each document's title and tags."

Step 3: Core Scoring Logic
Goal: Implement the main BM25 scoring algorithm. This function will be the heart of our relevance ranking.

AI Prompt:

"Generate a pure Python function calculate_bm25_scores(query_terms, search_context, config). It should implement the BM25 algorithm. Use the pre-computed inverted index and document lengths from the search_context, and the k1, b, and field_boosts parameters from the config dictionary."

Step 4: Post-Ranking for Diversity
Goal: Implement the MMR algorithm to re-rank the top results, ensuring they are diverse and not repetitive.

AI Prompt:

"Generate a pure Python function mmr_select(candidates, k, search_context, config). This function should implement Maximal Marginal Relevance. It will take a sorted list of candidate tuples (doc_id, score) and the number of results k. It should use the pre-computed TF-IDF vectors from the search_context to calculate cosine similarity for the redundancy check."

Step 5: Orchestration and Testing
Goal: Create the main search function that ties all the pieces together and write a set of tests to verify that the entire pipeline works as expected.

AI Prompt:

"Finally, create the main orchestrator function search(query, search_context, config). This function should call the helper functions in sequence: process the query, calculate scores, and then run MMR post-ranking. Also, generate a run_tests function that includes at least 3-4 assert statements to validate the end-to-end search logic for key queries, such as a basic keyword search, a synonym search, and a query that demonstrates MMR."
