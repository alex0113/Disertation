Next steps:

* use llm to convert news articles to knowledge graph format.
* test knowledge graph


*extract relevant knowledge from kg based on a fact 

1. Optimize Retrieval Strategies

Simply "fetching context" can lead to noise. You need to decide how you traverse the graph to get the most relevant information.

    Multi-hop Retrieval: Instead of just fetching the direct neighbors of a node, implement k-hop retrieval. This allows the LLM to understand distant relationships (e.g., "A is related to B, which is caused by C").

    Community Detection: Use algorithms like Leiden or Louvain to group nodes into communities. You can then summarize these communities to give the LLM a "high-level" view of the data before it dives into specific nodes.

    Sub-graph Extraction: Instead of raw triples, try passing a small, coherent subgraph to the LLM. This preserves the structural context better than a flat list of facts.


    2. Implement Advanced Reasoning Logic

Now that you have the data, how does the model "think" about it?

    Graph-of-Thought (GoT): Direct the LLM to navigate the graph iteratively. The model looks at a node, decides which edge to follow next, and builds its own path to the answer.

    Hybrid Search: Combine your KG retrieval with standard Vector Search (semantic search). KGs are great for structured facts; vectors are great for fuzzy, linguistic similarity. Seeing how they complement each other is excellent thesis material.


Metric	What it measures
Faithfulness	Does the LLM's answer actually come from the KG, or is it hallucinating?
Completeness	Did the KG retrieval find all relevant nodes needed to answer the question?
Robustness	How does the system handle "noise" or conflicting triples in the graph?
Comparison	How does your GraphRAG perform against a standard Vector-only RAG?



Entity Resolution: If your extractor finds "Einstein" and "Albert Einstein," can your system merge them automatically? This is a classic KG problem that adds significant "academic weight" to your work.



5. User Interface/Agentic Behavior

If your thesis has a practical component, consider turning your pipeline into a Graph-Agent. Instead of a static prompt, the agent could "query" the graph multiple times until it is satisfied it has the answer.



*find datasets

