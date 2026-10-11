# PowerPoint slides for the RAG workshop

**18 separate PowerPoint decks, 9 slides each, 162 slides total.** Each deck matches one notebook and includes concept explanations, worked examples, a live-demo walkthrough, an exercise, and presenter notes.

Download or copy all decks together: [RAG workshop presentations ZIP](RAG_workshop_presentations.zip). The PowerPoint files are self-contained; their cover and source images are embedded.

## Teaching with these slides

1. Open the deck and its matching notebook side by side.
2. Use PowerPoint **Notes** or **Presenter View** for explanations, demo prompts, and source references.
3. Run the notebook from a fresh kernel. Inspect retrieved evidence before showing a generated answer.
4. Use the exercise slide to change one variable and discuss the observed result.

Live notebook cells use your local Anthropic configuration. Offline rehearsal is explained in the project README. The slides do not call an API or contain your key. Examples use the fictional Northstar Analytics corpus.

## Deck index

| Notebook | PowerPoint | Slides |
|---|---|---:|
| 00 | [Your first Claude call](decks/00_setup_and_claude.pptx) | 9 |
| 01 | [Build RAG from scratch](decks/01_rag_from_scratch.pptx) | 9 |
| 02 | [Ingest files and design useful chunks](decks/02_ingestion_and_chunking.pptx) | 9 |
| 03 | [Embeddings and semantic search](decks/03_embeddings_and_semantic_search.pptx) | 9 |
| 04 | [Persistent vector stores and metadata](decks/04_vector_store_and_metadata.pptx) | 9 |
| 05 | [Compose RAG with LangChain](decks/05_langchain_rag.pptx) | 9 |
| 06 | [Hybrid search and reranking](decks/06_hybrid_search_and_reranking.pptx) | 9 |
| 07 | [Advanced retrieval](decks/07_advanced_retrieval.pptx) | 9 |
| 08 | [Conversational RAG](decks/08_conversational_rag.pptx) | 9 |
| 09 | [Evaluate RAG systematically](decks/09_evaluation_and_experiments.pptx) | 9 |
| 10 | [Agentic RAG](decks/10_agentic_rag.pptx) | 9 |
| 11 | [RAG production capstone](decks/11_production_capstone.pptx) | 9 |
| 12 | [Multimodal RAG](decks/12_multimodal_rag.pptx) | 9 |
| 13 | [Image and Document RAG](decks/13_image_document_rag.pptx) | 9 |
| 14 | [Audio RAG](decks/14_audio_rag.pptx) | 9 |
| 15 | [Video RAG](decks/15_video_rag.pptx) | 9 |
| 16 | [GraphRAG](decks/16_graph_rag.pptx) | 9 |
| 17 | [Structured-data RAG](decks/17_structured_data_rag.pptx) | 9 |

## Scope

The decks cover setup, text RAG, ingestion, embeddings, vector stores, LangChain, hybrid search, advanced retrieval, conversation, evaluation, agents, production considerations, multimodal RAG, images/documents, audio, video, graphs, and structured data.

The audio and video decks distinguish transcripts and selected frames from native audio/video model input. GraphRAG here is a compact source-backed knowledge graph. Structured-data RAG uses validated plans mapped to reviewed SQL queries.

## Sources and assets

Notebook section references and official documentation links are recorded in slide speaker notes. Slide text and tables remain editable. The chart and synthetic policy scan are original source evidence from the notebook data.

The shared cover background was created with the built-in Imagegen tool and saved in `assets/cover_background.png`. Prompt: "Create a refined 16:9 background image for a technical lecture series about retrieval augmented generation. No text, no letters, no numbers, no logos, no diagrams. Deep midnight navy throughout, calm large empty negative space across left 68% for future editable white titles. On the rightmost 30%, a tasteful photographic/stylized macro still life of semi-transparent layered glass sheets with subtle etched tiny lines suggesting stored knowledge, illuminated by one muted turquoise and soft violet edge light. Restrained, academic, premium editorial lighting, matte finish, uncluttered composition. Keep every point of brightness on right edge so left titles will have high contrast. Background asset only."
