import json
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from backend.config import settings
from backend.hybrid_retriever import HybridRetriever as ChromaRetriever
from backend.reranker import Reranker

logger = logging.getLogger(__name__)

# LLM Setup based on settings
if settings.use_groq:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0.7
    )
    logger.info(f"Using Groq model: {settings.model_name}")
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0.7
    )
    logger.info(f"Using OpenAI model: {settings.model_name}")


class RAGState(TypedDict):
    """State passed through the RAG pipeline."""
    query: str                      # original user query
    reformulated_query: str         # rewritten query for re-retrieval
    retrieved_chunks: list[dict]    # chunks from ChromaDB
    generated_answer: str           # LLM's answer
    critique: str                   # critic's verdict text
    is_grounded: bool               # True if answer passes critic
    retry_count: int                # how many retries so far
    final_answer: str               # what we return to user
    error: str                      # any error message
    conversation_history: list[dict] # previous user/assistant exchanges
    is_followup: bool               # whether this is a follow-up question
    collection_name: str            # active ChromaDB collection


def retrieve_node(state: RAGState) -> dict:
    """Retrieve relevant documents from ChromaDB."""
    retriever = ChromaRetriever(collection_name=state.get("collection_name", "rag_docs"))
    
    # Use reformulated query if retrying, otherwise use original query
    query_to_use = state["reformulated_query"] if state["retry_count"] > 0 else state["query"]
    
    try:
        chunks = retriever.retrieve(query_to_use, n_results=10)
        reranked_chunks = Reranker().rerank(query_to_use, chunks, top_k=3)
        return {"retrieved_chunks": reranked_chunks}
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {
            "retrieved_chunks": [],
            "error": f"Retrieval failed: {str(e)}"
        }


def generate_node(state: RAGState) -> dict:
    """Generate an answer based on retrieved context."""
    # Build context from chunks
    if not state["retrieved_chunks"]:
        answer = "No relevant documents found."
        return {"generated_answer": answer}
    
    context_parts = [chunk["text"] for chunk in state["retrieved_chunks"]]
    context = "\n---\n".join(context_parts)
    
    history = state.get("conversation_history", [])
    if history:
        formatted_history = "\n".join(
            f"{'User' if item.get('role') == 'user' else 'Assistant'}: {item.get('content', '')}"
            for item in history
        )
        prompt = f"""Previous conversation:
{formatted_history}

Now answer the new question based on context and conversation history. Use the retrieved context as the primary source.

Context: 
{context}

Question: {state["query"]}

Answer:"""
    else:
        prompt = f"""Answer the question based ONLY on the context below.

Context: 
{context}

Question: {state["query"]}

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"generated_answer": answer}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            "generated_answer": "",
            "error": f"Generation failed: {str(e)}"
        }


def critic_node(state: RAGState) -> dict:
    """Critique the generated answer for groundedness."""
    if not state["generated_answer"]:
        return {
            "is_grounded": False,
            "critique": "No answer was generated",
            "reformulated_query": state["query"],
            "retry_count": state["retry_count"] + 1
        }
    
    # Build context from chunks
    context_parts = [chunk["text"] for chunk in state["retrieved_chunks"]]
    context = "\n---\n".join(context_parts) if context_parts else "No context available"
    
    # Build critic prompt
    critic_prompt = f"""You are a grounding critic. Given a question, context chunks, and an answer, determine if the answer is grounded in the context.

Question: {state["query"]}

Context: 
{context}

Answer: {state["generated_answer"]}

Reply with ONLY a JSON object like this (no markdown, no extra text):
{{"grounded": true, "reason": "...", "reformulated_query": "..."}}

Set grounded=true if the answer is supported by context.
Set grounded=false if the answer contains hallucinations or info not in context.
reformulated_query should be a better search query to find missing info (used only if grounded=false)."""
    
    try:
        response = llm.invoke(critic_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        critique_data = json.loads(response_text)
        
        return {
            "is_grounded": critique_data.get("grounded", False),
            "critique": critique_data.get("reason", ""),
            "reformulated_query": critique_data.get("reformulated_query", state["query"]),
            "retry_count": state["retry_count"] + 1
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse critic response: {e}")
        return {
            "is_grounded": False,
            "critique": f"Critic parsing error: {str(e)}",
            "reformulated_query": state["query"],
            "retry_count": state["retry_count"] + 1,
            "error": f"Critic JSON parsing failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Critic error: {e}")
        return {
            "is_grounded": False,
            "critique": f"Critic error: {str(e)}",
            "reformulated_query": state["query"],
            "retry_count": state["retry_count"] + 1,
            "error": f"Critic failed: {str(e)}"
        }


def finalize_node(state: RAGState) -> dict:
    """Finalize the answer or indicate inability to answer."""
    if state["is_grounded"]:
        return {"final_answer": state["generated_answer"]}
    
    if state["retry_count"] >= settings.max_retries:
        return {
            "final_answer": "I don't have enough information to answer this accurately based on the available documents."
        }
    
    # Continue looping (will be caught by conditional edge)
    return {}


def should_continue(state: RAGState) -> str:
    """Determine whether to continue retrying or finalize."""
    if state["is_grounded"]:
        return "finalize"
    
    if state["retry_count"] >= settings.max_retries:
        return "finalize"
    
    return "retrieve"


def build_graph():
    """Build and compile the RAG pipeline graph."""
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critic")
    
    # Conditional edge from critic
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "retrieve": "retrieve",
            "finalize": "finalize"
        }
    )
    
    # Exit edge
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


def run_pipeline(
    query: str,
    conversation_history: list[dict] = None,
    collection_name: str = "rag_docs"
) -> dict:
    """
    Run the complete RAG pipeline.
    
    Args:
        query: The user's question
        
    Returns:
        Dict with final_answer, critique, retry_count, retrieved_chunks (texts only)
    """
    graph = build_graph()
    
    if conversation_history is None:
        conversation_history = []

    # Create initial state
    initial_state = {
        "query": query,
        "reformulated_query": "",
        "retrieved_chunks": [],
        "generated_answer": "",
        "critique": "",
        "is_grounded": False,
        "retry_count": 0,
        "final_answer": "",
        "error": "",
        "conversation_history": conversation_history,
        "is_followup": bool(conversation_history),
        "collection_name": collection_name
    }
    
    # Run the pipeline
    result = graph.invoke(initial_state)
    
    final_answer = result["final_answer"]
    updated_history = conversation_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": final_answer}
    ]

    return {
        "final_answer": final_answer,
        "critique": result["critique"],
        "retry_count": result["retry_count"],
        "retrieved_chunks": [
            {
                "text": c["text"] if isinstance(c, dict) else c,
                "semantic_score": round(c.get("semantic_score", 0), 3) if isinstance(c, dict) else 0,
                "bm25_score": round(c.get("bm25_score", 0), 3) if isinstance(c, dict) else 0,
                "final_score": round(c.get("final_score", 0), 3) if isinstance(c, dict) else 0,
                "rerank_score": round(
                    min(max((c.get("rerank_score", 0) + 10) / 20, 0), 1), 3
                ) if isinstance(c, dict) else 0,
                "metadata": c.get("metadata", {}) if isinstance(c, dict) else {}
            }
            for c in result.get("retrieved_chunks", [])
        ],
        "is_grounded": bool(result.get("is_grounded", False)),
        "conversation_history": updated_history,
    }
