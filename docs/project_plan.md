Part 1: Project Brief
This section outlines the high-level plan for our rapid search engine prototype.

Goal 🎯
To build a functional, feature-based search ranking model in a single Python script within one hour. The model will rank design templates based on a combination of keyword relevance, semantic similarity, and business metrics.

Project Scope
We will implement an end-to-end pipeline that includes: mock data generation, feature creation, model training with XGBRanker, and a search function to rank results for a given query.

Core Features ✨

Mock Data Generation: Create a realistic but simple dataset of templates and user search interactions.

Hybrid Feature Engineering: Generate features combining BM25 (keyword), sentence embeddings (semantic), and business logic (popularity).

Gradient-Boosted Ranking: Train and use an XGBRanker model to predict relevance scores.

Constraints 🚧
To align with the project's rapid, AI-assisted nature, we will adhere to the following constraints:

AI-Driven Development: Each step is designed as a clear instruction for an AI coding assistant to generate the necessary code blocks.

Single File: The complete, runnable solution will be contained in one .py file for simplicity.

Step 1: Scaffolding and Mock Data Construction
Goal: Generate all necessary mock data structures to simulate a real-world template library and user search logs. This step should be almost entirely handled by the AI assistant based on specific instructions.

Action Plan (Instructions for AI Assistant):

Create Template Database: Generate a list of 10-15 Python dictionaries representing our design templates. Each dictionary must have the keys: template_id (e.g., t001), title (e.g., "Modern Business Card"), description (a short sentence), and popularity_score (a random integer between 100 and 5000).

Construct Training Search Logs: This is the most critical part. Create a list of dictionaries representing user search interactions. This list will be used to train our model. Each dictionary should have three keys:

query: The text the user searched for (e.g., "professional resume").

template_id: The ID of a template that was shown for that query.

relevance: A score indicating how relevant that template was. This is the target we will predict.

How to create realistic relevance scores: For each unique query (e.g., "birthday invite"), create 4-5 entries.

Assign a relevance of 3 to one template that is a perfect match (e.g., a template titled "Kids Birthday Party Invite").

Assign a relevance of 2 to one template that is a good semantic match (e.g., "Fun Party Announcement").

Assign a relevance of 1 or 0 to the remaining templates that are poor matches (e.g., "Corporate Business Card"). This teaches the model what not to rank highly.

Step 2: Feature Generation
Goal: Create a single, pure function that takes the raw data from Step 1 and generates a complete feature matrix (X) and a relevance scores vector (y). This function is the core of our data processing pipeline.

Action Plan (Instructions for AI Assistant):

Define the Feature Generation Function: Create a function generate_features(search_logs, template_db).

Merge Data: Inside the function, convert the search_logs and template_db into pandas DataFrames and merge them so each row represents one (query, template) pair with all its associated data.

Create Keyword Feature (BM25): For each unique query, calculate the BM25 score for all templates against that query. Add this score as a new column named bm25_score.

Create Semantic Feature (Embeddings):

Use the sentence-transformers library to encode all unique queries and all template descriptions into embeddings.

For each (query, template) pair in your DataFrame, calculate the cosine similarity between their respective embeddings.

Add this similarity score as a new column named embedding_similarity.

Assemble Final Feature Matrix: The function should return three objects:

X: A DataFrame containing only the feature columns (bm25_score, embedding_similarity, popularity_score).

y: A Series containing the relevance scores.

groups: A list or array indicating the size of each query group. For example, if the first query has 4 associated templates and the second has 5, the groups array would be [4, 5, ...]. This is essential for XGBRanker.

Step 3: Model Training and Ranking Logic
Goal: Train the XGBRanker model on the generated features and create a simple function that uses the trained model to rank templates for a new query.

Action Plan (Instructions for AI Assistant):

Train the XGBRanker Model:

Create an XGBRanker model object with objective='rank:ndcg'.

Create an XGBoost DMatrix from the features X, labels y, and the groups array generated in Step 2.

Train the model using xgb.train().

Create the Ranking Function: Create a function rank_templates(query, model, template_db).

Inside the Ranking Function:

This function will take a new user query as input.

It must generate the same feature vector for the input query against all templates in our template_db.

Use the trained model.predict() method on this feature vector to get a relevance score for each template.

Sort the templates by this predicted score in descending order and return the sorted list.

Step 4: Orchestration and Testing
Goal: Create a main execution block that ties all the steps together and runs a test search to verify the entire pipeline works as expected.

Action Plan (Instructions for AI Assistant):

Create a main block: Use an if __name__ == "__main__": block to orchestrate the script.

Call the Functions in Order:

Call the data generation functions from Step 1.

Call the generate_features function from Step 2 to get X, y, and groups.

Train the XGBRanker model as described in Step 3.

Call the rank_templates function with a test query (e.g., "professional resume").

Print Results: Print the ranked list of template titles from the test query. Verify that the results are more relevant than a simple keyword search would provide. For example, a template titled "Modern CV Design" should rank highly for the "professional resume" query, demonstrating semantic understanding.
