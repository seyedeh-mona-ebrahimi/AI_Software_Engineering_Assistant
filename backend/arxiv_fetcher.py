import requests
from bs4 import BeautifulSoup


def fetch_latest_arxiv():
    """Fetches latest AI and Software Engineering research papers from ArXiv."""
    urls = [
        "https://arxiv.org/list/cs.AI/recent",   # Artificial Intelligence
        "https://arxiv.org/list/cs.SE/recent"     # Software Engineering
    ]
    papers = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for entry in soup.find_all("div", class_="list-title", limit=5):
            # Change "title" to "text" for consistency
            text = entry.text.replace("Title:", "").strip()
            link_tag = entry.find_previous("a")
            if link_tag:
                link = "https://arxiv.org" + link_tag["href"]
                papers.append({"text": text, "source": "ArXiv", "link": link})
    return papers


def fetch_latest_github():
    url = "https://api.github.com/search/repositories?q=ci%2Fcd+in:readme&sort=stars&order=desc"
    response = requests.get(url)
    data = response.json()
    repos = []
    for repo in data.get("items", [])[:5]:
        repos.append({
            "text": repo["description"] or repo["name"],
            "source": "GitHub",
            "link": repo["html_url"]
        })
    return repos



def fetch_latest_stackoverflow():
    url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=ci-cd&site=stackoverflow"
    response = requests.get(url)
    data = response.json()
    questions = []
    for item in data.get("items", [])[:5]:
        questions.append({
            "text": item["title"],
            "source": "Stack Overflow",
            "link": item["link"]
        })
    return questions




def fetch_latest_rfc():
    """Fetches latest RFCs related to AI and Software Engineering."""
    url = "https://www.rfc-editor.org/search/rfc_search_detail.php?title=AI+OR+Software"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    rfcs = []
    for row in soup.find_all("tr")[1:6]:  # Skip the header row
        cols = row.find_all("td")
        if len(cols) >= 2:
            text = cols[1].text.strip()
            link_tag = cols[1].find("a")
            if link_tag:
                link = "https://www.rfc-editor.org" + link_tag["href"]
                rfcs.append({"text": text, "source": "RFCs", "link": link})
    return rfcs

def fetch_latest_articles():
    urls = ["https://arxiv.org/list/cs.AI/recent", "https://arxiv.org/list/cs.SE/recent"]
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for entry in soup.find_all("div", class_="list-title", limit=5):
            text = entry.text.replace("Title:", "").strip()
            link_tag = entry.find_previous("a")
            link = "https://arxiv.org" + link_tag["href"] if link_tag else "#"
            # Optionally: attempt to get the abstract if available
            articles.append({"text": text, "source": "ArXiv", "link": link})
    return articles




def fetch_all_sources():
    arxiv= fetch_latest_arxiv()
    github_results = fetch_latest_github()
    so_results = fetch_latest_stackoverflow()
    arxiv_results = fetch_latest_articles()  # or fetch_latest_arxiv()
    rfc_results = fetch_latest_rfc()
    return github_results + so_results + arxiv_results + rfc_results + arxiv




if __name__ == "__main__":
    latest_sources = fetch_all_sources()
    for article in latest_sources:
        print(f"🔹 {article['title']} ({article['source']})\n🔗 {article['link']}\n")


