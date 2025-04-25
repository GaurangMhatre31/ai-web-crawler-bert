import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Step 1: Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# Step 2: Define state
class AgentState(TypedDict):
    question: str
    urls: Annotated[List[str], "List of URLs to crawl"]
    research_data: Annotated[List[str], "Scraped content"]
    answer: Annotated[str, "Final answer"]

# Step 3: Scraping Agent
def research_agent(state: AgentState) -> AgentState:
    scraped_data = []
    for url in state["urls"]:
        try:
            print(f"[Scraping] {url}")
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            scraped_data.append(" ".join(paragraphs))
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            continue
    return {**state, "research_data": scraped_data}

# Step 4: Draft Agent
def generate_answer(state: AgentState) -> AgentState:
    context = " ".join(state["research_data"])[:512]  # truncate for BERT
    question = state["question"]
    inputs = tokenizer.encode_plus(question + " " + context, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_index = torch.argmax(outputs.logits, dim=-1)
    answer = tokenizer.decode(predicted_index[0], skip_special_tokens=True)
    return {**state, "answer": answer}

# Step 5: LangGraph setup
workflow = StateGraph(AgentState)
workflow.add_node("research", research_agent)
workflow.add_node("draft", generate_answer)
workflow.set_entry_point("research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", END)
app_graph = workflow.compile()

# Step 6: Run CLI
if __name__ == "__main__":
    print("=== Local Web Crawler + BERT QA ===")
    question = input("Enter your question: ")
    print("Enter URLs (comma separated):")
    url_input = input("> ")
    urls = [u.strip() for u in url_input.split(",") if u.strip()]
    final = app_graph.invoke({"question": question, "urls": urls})
    print("\n========= Final Answer =========\n")
    print(final["answer"])
    print("\n================================\n")
