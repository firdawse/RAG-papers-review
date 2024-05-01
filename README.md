
# RAG Papers Review

- Table of Contents:
  - [What is a RAG System?](#what-is-a-rag-system)
  - [How Can We Evaluate a RAG System?](#how-can-we-evaluate-a-rag-system)
  - [Existing RAG Benchmarks](#existing-rag-benchmarks)
  - [How to Improve a RAG System](#how-to-improve-a-rag-system)
    - [Pre-retrieval Techniques](#pre-retrieval-techniques)
    - [Retrieval Techniques](#retrieval-techniques)
    - [Post-Retrieval Techniques](#post-retrieval-techniques)
  - [Advanced RAG Techniques](#advanced-rag-techniques)
    - [SELF RAG](#self-rag)
    - [Corrective RAG](#corrective-rag)
    - [RAT (Retrieval Augmented Thoughts)](#rat-retrieval-augmented-thoughts)
  - [Insight from ARAGOG Study](#insight-from-aragog-study)
  - [References](#references)

## What is a RAG System?
Retrieval augmented generation (RAG) is a strategy designed to address both LLM hallucinations and outdated training data. The RAG workflow consists of three key steps. First, the corpus is divided into discrete chunks, and vector indices are created using an encoder model. Second, RAG identifies and retrieves chunks based on their vector similarity to the query and indexed chunks. Finally, the model generates a response conditioned on the contextual information obtained from the retrieved chunks. RAG enables language models to circumvent the need for retraining, allowing access to the latest information for generating reliable outputs through retrieval-based generation.

![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled.png)

Source: [2404.10981.pdf (arxiv.org)](https://arxiv.org/pdf/2404.10981.pdf)

### How Can We Evaluate a RAG System?

Evaluating language models (LM) continues to pose challenges for many. Whether you're refining a model's accuracy through fine-tuning or enhancing a Retrieval-Augmented Generation (RAG) system's contextual relevancy, grasping the process of developing and selecting the right LLM evaluation metrics for your specific use case is crucial for establishing a robust LLM evaluation pipeline.

This taxonomy attempts to combine and organize metrics and tools used to evaluate RAG systems based on reviews from several papers.

![Untitled](1.png)

RAG evaluation taxonomy can be divided into two primary targets: retrieval quality and generation quality.

For retrieval quality, the focus is on how relevant the retrieved documents are to the query. This aspect can be measured based on some order-anware metrics that are indifferent to the ranking of retrieved documents, such as:

- **RECALL@k**: Measures the proportion of correctly identified relevant items in the top K recommendations out of the total number of relevant items in the dataset.
- **Precision@k**: Measures the number of relevant items over the total number of items.
- **F1@K**: Accounts for both precision and recall.
- **hit rate@k**: Counts how many times at least one relevant document was retrieved within the top K across all queries, and it increases with K.

Then, there are other metrics that can be categorized as order-aware since they consider not only the relevance of retrieved items but also their relative rank, such as:

- **MRR@k (Mean Reciprocal Rank)**: Shows how soon you can find the first relevant item. It's beneficial when the top-ranked item matters but disregards all the remaining items and is less informative for situations where multiple highly relevant documents are retrieved.
- **MAP (Mean Average Precision)**: Takes into consideration the entire retrieved list and evaluates the average precision at all the relevant ranks within the top K.
- **NDCG (Normalized Discounted Cumulative Gain)**: Tries to evaluate the match with the ideal order, comparing ranking to an ideal order where all the relevant items are at the top of the list. It calculates the DCG, which is the sum of relevant items over the log of its rank, normalized by the Ideal NGC where we put all the relevant items at the top. It's based on the idea that items that are higher in the rank should be given more credit. It's similar to MAP but more sensitive to rank order.

Context relevance can also be measured by LLM. RAGAS, Trulens, and ARES in general are some open-source frameworks for evaluating RAG pipelines based on LLM reasoning guided by a specific prompt.

The second part of the taxonomy focuses on generation quality. Here, we evaluate aspects such as faithfulness or groundedness, which determine how well the final response aligns with the retrieved information. These evaluations can be conducted using frameworks like RAGAS, Trulens, and ARES.

Moving on, we have answer relevance, which assesses whether your language model (LM) outputs concise and pertinent answers aligned with the query. This assessment can be measured against **ground truth** using quantitative metrics such as:

- Cosine similarity
- BLEU: Based on precision n-grams but does not consider recall. It assigns equal importance to propositions alongside nouns and verbs.
- Rouge, essentially akin to calculating the F1 score, also based on overlapping n-grams.
- METEOR, somewhat similar to Rouge, as it considers both precision and recall but with different weightings. It also accounts for words with similar meanings.

Transitioning to model-based metrics:

- Prometheus: LLM fine-tuned for evaluation, requires reference and example evaluation requirements. It's open-source, pretrained, and generates scores based on your specified requirements, along with a rationale for the score.

![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%202.png)

- BERTScore: Focuses on the similarity between reference and prediction. It starts by generating BERT embeddings for both reference and generated answers, computes pairwise cosine similarity, creates a similarity matrix, sums the maximum similarities, and then normalizes to derive recall and precision scores before calculating the F1 BERTScore.
- MoverScore: Unlike BERTScore's hard alignment, relies on soft alignment, allowing for many-to-one mappings of semantically related words between two sequences. This approach minimizes the effort needed to transform between texts, utilizing the Earth Mover's Distance to compute the minimal cost required to transform word distributions between an LM output and a reference text.

For evaluating answers without ground truth:

- G-eval: Prompts an LM to follow a reasoning chain to evaluate text against specific user-specified criteria, generating a score at the end.

![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%203.png)

- **Self-Check GPT:** Employs a sample-based approach, assuming that hallucinated outputs are not reproducible. It generates multiple answers and evaluates each sample against

 the responses to determine if they are supported by all generated sentences, resulting in an accuracy score. This approach is suitable only for hallucination evaluation, not for other cases.

![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%204.png)

Additionally, specific abilities can be evaluated, such as noise robustness, calculated using accuracy or precision, negative rejection, evaluated in RGB benchmarks based on EM, information integration assessed using EM scores, and counterfactual robustness measured using **R-rate**, which is the proportion of edited words appearing in the model’s outputs in all edited words for text generation task, and **M-rate**, which is the proportion of the queries that the model answers wrongly with edited contexts in all queries that the model can answer correctly without external knowledge for question-answer tasks.

## Existing RAG Benchmarks:

| Benchmark | Code Available | Dataset | Task | Abilities Evaluated | Metrics | Mitigating Methods | Phase Evaluated | Generation Models | Retrieval Models | Weaknesses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2309.01431.pdf](https://arxiv.org/pdf/2309.01431.pdf) | [Code](https://github.com/chen700564/RGB) | [Dataset](https://github.com/chen700564/RGB) | Question Answering | Noise robustness, negative rejection, information integration, counterfactual robustness | Accuracy Rejection rate Error detection rate Error correction rate | N/A | Generation | ChatGPT ChatGLM-6B ChatGLM2-6B Vicuna-7B-v1.3 Qwen-7B-Chat BELLE-7B-2M | | - Error analysis lacking evidence - No additional approaches were evaluated. |
| [2311.08147.pdf](https://arxiv.org/pdf/2311.08147.pdf) | No | No | Question Answering Text Generation | Counterfactual robustness | Accuracy BLEU ROUGE-L Misleading Rate (M-Rate) Mistake Reappearance Rate (R-Rate) | Prompting DOLA | Generation | ChatGLM2 Llama2 Vicuna Baichuan2 | | - No available source code. - No explanation of the choice of the mitigating techniques. - Experiments based on only 3 random seeds. - Resources allocated not mentioned. |
| [2401.15391](https://arxiv.org/abs/2401.15391) | [Code](https://github.com/yixuantt/) | [Dataset](https://github.com/yixuantt/) | Question Answering | Information integration | Mean Average Precision at K (MAP@K) Hit Rate at K (Hit@K) Mean Reciprocal Rank at K (MRR@K) | Reranker | Retrieval Generation | GPT-4 ChatGPT Llama-2-70b-chat-hf Mixtral-8x7B-Instruct Claude-2.1 Google-PaLM | text-embedding ada-002 text-search ada-query-001 llm-embedder bge-large-en-v1.5 jina-embeddings-v2-base-en intfloat/e5-base-v2 voyage-02 hkunlp/instructor-large | - Resources allocated not mentioned. - Additional approaches could be evaluated |

## How to Improve a RAG System:

Various techniques can be applied to enhance each component of the RAG workflow:

### Pre-retrieval Techniques:

The performance of your RAG solution depends on how well the data is cleaned and organized. That’s why the first thing to consider is cleaning the data, removing unnecessary words, tags, or noisy data. Then you have to choose the right size and way of chunking your data. If your chunk is too small, it may not include all the information the LLM needs to answer the user’s query; if the chunk is too big, it may contain too much irrelevant information that confuses the LLM or may be too big to fit into the context size.

- **Chunking Techniques:**

1. **Sentence Window Retrieval:**

    This process involves embedding a limited set of sentences for retrieval, with the additional context surrounding these sentences, referred to as “window context,” stored separately and linked to them. Once the top similar sentences are identified, this context is reintegrated just before these sentences are sent to the Large Language Model (LLM) for generation, thereby enriching overall contextual comprehension.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%205.png)

1. **Parent Document Retrieval:**

    During retrieval, it first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents to the LLM. It utilizes small text blocks during the initial search phase and subsequently provides larger related text blocks to the language model for processing.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%206.png)

1. **Semantic Chunking:**

    Semantic Chunking considers the relationships within the text. It divides the text into meaningful, semantically complete chunks. This approach ensures the information's integrity during retrieval, leading to a more accurate and contextually appropriate outcome.

    Semantic chunking involves taking the embeddings of every sentence in the document, comparing the similarity of all sentences with each other, and then grouping sentences with the most similar embeddings together.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%207.png)

### Retrieval Techniques:

1. **Hyde:**

    Hyde is an approach that uses a Language Learning Model (similar to ChatGPT) to generate a theoretical document when answering a question. Instead of directly searching for answers in a database.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%208.png)

    Image Source: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496.pdf)

1. **Hybrid Search:**

    Hybrid search is a technique that combines multiple search algorithms to improve the accuracy and relevance of search results. It uses the best features of both keyword-based search algorithms with vector search techniques. By leveraging the strengths of different algorithms, it provides a more effective search experience for users.

1. **StepBack-prompt:**

    This approach encourages the language model to think beyond specific examples and focus on broader concepts and principles.

    This template replicates the “Step-Back” prompting technique thatimproves performance on complex questions by first asking a “step back” question. This technique can be combined with standard question-answering RAG applications by retrieving information for both the original and step-back questions. Below is an example of a step-back prompt.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%209.png)

    Image Source: [TAKE A STEP BACK: EVOKING REASONING VIA ABSTRACTION IN LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.06117.pdf)

1. **Multiquery Retriever:**

    The Multi-query Retrieval method utilizes LLMs to generate multiple queries from different perspectives for a given user input query, advantageous for addressing complex problems with multiple sub-problems. For each query, it retrieves a set of relevant documents to get a larger set of potentially relevant documents. By generating multiple perspectives on the same question, the MultiQuery Retriever might be able to overcome some of the limitations of the distance-based retrieval and get a richer set of results.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2010.png)

### Post-Retrieval Techniques:

- **Re-Ranking:**

    After the retriever generates a set of potential candidates from a large document collection, the reranker evaluates these candidates more thoroughly. A reranking model — also known as a cross-encoder — is a type of model that, given a query and document pair, will output a similarity score. We use this score to reorder the documents by relevance to our query. It’s much more accurate than embedding models.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2011.png)

    Image source: [Pinecone](https://www.pinecone.io/learn/series/rag/rerankers/)

- **Contextual Compression:**

    Noise in retrieved documents adversely affects RAG performance. Therefore, the most relevant information to a query may be buried in a document with a lot of irrelevant text. Contextual compression is meant to fix this. The idea is instead of immediately returning retrieved documents as-is, it can compress them using the Doc Compressor, a small language models to calculate prompt mutual information of the user query and retrieved document, estimating element importance.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2012.png)

- **Maximal Marginal Relevance (MMR):**

    This technique aims to refine the retrieval process by striking a balance between relevance and diversity in the documents retrieved. By employing MMR, the retrieval system evaluates potential documents not only for their closeness

## Advanced RAG Techniques:

- **SELF RAG:**

    The SELF-RAG framework trains a single arbitrary language model (llama 2) to adaptively retrieve passages on-demand. To generate and reflect on retrieved passages and on own generations using special tokens, called reflection tokens. These reflection tokens signal the need for retrieval or confirm the output’s relevance, support, or completeness.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2013.png)

- **Corrective RAG:**

    Given an input query and the retrieved documents from any retriever, a lightweight retrieval evaluator is constructed to estimate the relevance score of retrieved documents to the input query. The relevance score is quantified into three confidence degrees: Correct, Incorrect, or Ambiguous.

    1. **Correct**: If this action is triggered, the retrieved documents are refined into more precise knowledge strips. This refinement operation involves knowledge decomposition, filtering, and recomposition.
    2. **Incorrect**: If this action is triggered, the retrieved documents are discarded. Instead, web searches are used as complementary knowledge sources for corrections.
    3. **Ambiguous**: When it cannot confidently make a correct or incorrect judgment, a soft and balanced action is triggered, combining elements of both Correct and Incorrect judgments.

    After optimizing the retrieval results, an arbitrary generative model can be adopted.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2014.png)

- **RAT (Retrieval Augmented Thoughts):**

    **Retrieval Augmented Thoughts (RAT)** is a simple yet effective prompting strategy that combines Chain-of-Thought (CoT) prompting and retrieval augmented generation (RAG) to address long-horizon reasoning and generation tasks.

    **Step One:** Firstly, the initial zero-shot CoT prompt produced by LLMs along with the original task prompt are used as queries to retrieve the information that could help revise the possibly flawed CoT.

    **Step Two:** Secondly, instead of retrieving and revising with the full CoT and producing the final response at once, a progressive approach, where LLMs produce the response step-by-step following the CoT (a series of sub-tasks).

    Only the current thought step is revised based on the information retrieved with the task prompt, the current, and the past CoTs.

    ![Untitled](RAG%20Papers%20review%2078ff49fce0a849b4bf3f8fcfd80bbe97/Untitled%2015.png)

## Insight from ARAGOG Study:
The ARAGOG study carefully assessed various RAG techniques, using metrics such as Retrieval Precision and Answer Similarity to evaluate their effectiveness. Among the techniques examined, Hypothetical Document Embedding (HyDE) and LLM reranking stood out as methods significantly enhancing retrieval precision. However, these methods require additional LLM queries, leading to increased latency and cost. On the contrary, Maximal Marginal Relevance (MMR) and Cohere rerank showed limited benefits, and Multi-query approaches performed worse than a basic Naive RAG system.

Interestingly, Sentence Window Retrieval emerged as highly effective in retrieval precision, despite showing variable performance in answer similarity. This suggests its potential, though further investigation is needed to fully exploit its capabilities. Additionally, the study highlighted the Document Summary Index method as a promising retrieval approach, indicating its future potential with further enhancements.

### References
- https://arxiv.org/pdf/2404.01037.pdf
- https://arxiv.org/pdf/2310.11511.pdf
- https://arxiv.org/pdf/2310.06117.pdf
- https://arxiv.org/pdf/2212.10496.pdf
- https://arxiv.org/pdf/2312.10997.pdf
- https://arxiv.org/pdf/2401.15391.pdf
- https://arxiv.org/pdf/2401.15884.pdf
- https://arxiv.org/pdf/2309.01431.pdf
- https://arxiv.org/pdf/2403.05313.pdf
- https://www.promptingguide.ai/techniques/rag

