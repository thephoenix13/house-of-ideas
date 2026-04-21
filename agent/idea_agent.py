import os
import json
import uuid
from datetime import date
from pathlib import Path
from anthropic import Anthropic
from tavily import TavilyClient
from dotenv import load_dotenv
from domains import DOMAINS

load_dotenv()

anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

OUTPUT_DIR = Path(__file__).parent / "output" / "ideas"
INDEX_PATH = Path(__file__).parent / "output" / "index.json"

SYSTEM_PROMPT = """You are the House of Ideas research analyst. House of Ideas is a venture studio that funds and mentors students from India's premier institutions (IITs, NITs) to build high-potential startups.

Your job: analyse web search results and extract high-potential startup ideas suitable for the HoI portfolio.

A good HoI idea must be:
- Specific and actionable — not a generic category
- Executable by a small team with early-stage capital
- Relevant to the Indian market
- Timely — there is a clear reason why now

For each idea, return a JSON object with these exact fields:
{
  "idea_title": "A clear, specific name — not generic",
  "problem_statement": "The specific, verifiable pain point",
  "market_opportunity": "TAM estimate and growth signals",
  "why_now": "What has changed recently that makes this timely",
  "competitive_landscape": "Who exists, what is the gap",
  "india_fit": "Relevance in Indian regulatory and cultural context",
  "execution_feasibility": "What a small team can achieve in 6 months with early capital",
  "hoi_score": <integer 1-10 on overall HoI portfolio fit>,
  "sources": ["url1", "url2"]
}

Return a JSON array of exactly 2-3 ideas. No markdown, no explanation — only the JSON array."""


def search_domain(domain_name, queries):
    all_results = []
    for query in queries:
        try:
            response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )
            all_results.extend(response.get("results", []))
        except Exception as e:
            print(f"    Search error for '{query}': {e}")
    return all_results


def extract_ideas(domain_name, search_results):
    results_text = "\n\n".join([
        f"URL: {r.get('url', '')}\nTitle: {r.get('title', '')}\nSummary: {r.get('content', '')}"
        for r in search_results[:15]
    ])

    response = anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Domain: {domain_name}\n\nSearch results:\n{results_text}\n\nReturn 2-3 ideas as a JSON array.",
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def save_ideas(domain_name, ideas):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for idea in ideas:
        idea_id = str(uuid.uuid4())[:8]
        idea["id"] = idea_id
        idea["domain"] = domain_name
        idea["date_discovered"] = date.today().isoformat()
        idea["status"] = "pending_review"

        filepath = OUTPUT_DIR / f"{idea_id}.json"
        with open(filepath, "w") as f:
            json.dump(idea, f, indent=2)

        saved.append({
            "id": idea_id,
            "title": idea.get("idea_title"),
            "domain": domain_name,
            "hoi_score": idea.get("hoi_score"),
            "file": str(filepath),
        })
        print(f"    [{idea.get('hoi_score')}/10] {idea.get('idea_title')}")

    return saved


def update_index(new_entries):
    existing = []
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            existing = json.load(f)

    existing.extend(new_entries)
    existing.sort(key=lambda x: x.get("hoi_score") or 0, reverse=True)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def run():
    print("House of Ideas — Idea Discovery Agent")
    print("=" * 42)
    print(f"Scanning {len(DOMAINS)} domains...\n")

    all_saved = []

    for domain in DOMAINS:
        print(f"Domain: {domain['name']}")
        results = search_domain(domain["name"], domain["queries"])
        print(f"  {len(results)} search results retrieved")

        if not results:
            print("  No results — skipping\n")
            continue

        try:
            ideas = extract_ideas(domain["name"], results)
            saved = save_ideas(domain["name"], ideas)
            all_saved.extend(saved)
        except Exception as e:
            print(f"  Error: {e}")

        print()

    update_index(all_saved)

    print("=" * 42)
    print(f"Done. {len(all_saved)} ideas discovered.")
    print(f"Index: {INDEX_PATH}")
    print(f"Ideas: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
