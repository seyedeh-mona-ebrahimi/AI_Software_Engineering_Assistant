import requests
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib
import json
import scholarly
# ---------------------------
# Utility Functions
# ---------------------------
def generate_doc_id(text, source):
    return hashlib.md5(f"{source}:{text}".encode()).hexdigest()

def format_doc(text, source, link):
    return {
        "id": generate_doc_id(text, source),
        "text": text,
        "source": source,
        "link": link,
        "timestamp": datetime.utcnow().isoformat()
    }

# ---------------------------
# ArXiv Fetcher
# ---------------------------
def fetch_latest_arxiv():
    urls = [
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/list/cs.SE/recent"
    ]
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for entry in soup.find_all("div", class_="list-title", limit=5):
            title = entry.text.replace("Title:", "").strip()
            link_tag = entry.find_previous("a")
            link = "https://arxiv.org" + link_tag["href"] if link_tag else "#"
            articles.append(format_doc(title, "ArXiv", link))
    return articles

# ---------------------------
# GitHub Fetcher
# ---------------------------
def fetch_latest_github():
    headers = {
        "Accept": "application/vnd.github.v3+json",
        # Replace with your GitHub token if needed
        # "Authorization": "Bearer YOUR_TOKEN"
    }
    url = "https://api.github.com/search/repositories?q=ci/cd+in:readme&sort=stars&order=desc"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("GitHub API error:", response.status_code, response.text)
        return []

    data = response.json()
    repos = []
    for repo in data.get("items", [])[:5]:
        desc = repo.get("description") or repo.get("full_name")
        repos.append(format_doc(desc, "GitHub", repo.get("html_url")))
    return repos

# ---------------------------
# Stack Overflow Fetcher
# ---------------------------
def fetch_latest_stackoverflow():
    url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=devops&site=stackoverflow"
    response = requests.get(url)
    if response.status_code != 200:
        print("Stack Overflow API error:", response.status_code, response.text)
        return []

    data = response.json()
    questions = []
    for item in data.get("items", [])[:5]:
        questions.append(format_doc(item.get("title"), "Stack Overflow", item.get("link")))
    if not questions:
        print("Stack Overflow returned no results.")
    return questions

# ---------------------------
# RFC Fetcher
# ---------------------------
def fetch_latest_rfc():
    url = "https://www.rfc-editor.org/search/rfc_search_detail.php?title=software&sortkey=Number&sorting=DESC"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    rfcs = []
    table = soup.find("table", class_="gridtable")
    if not table:
        print("RFC table not found.")
        return []

    rows = table.find_all("tr")[1:6]
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 3:
            rfc_number = cols[0].text.strip()
            title = cols[2].text.strip()
            link_tag = cols[0].find("a")
            link = "https://www.rfc-editor.org" + link_tag["href"] if link_tag else "#"
            link = link.replace("https://www.rfc-editor.orghttps://www.rfc-editor.org", "https://www.rfc-editor.org")
            rfcs.append(format_doc(f"{rfc_number}: {title}", "RFC", link))
    return rfcs

# ---------------------------
# Hacker News Fetcher
# ---------------------------
def fetch_latest_hackernews():
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    if response.status_code != 200:
        print("Hacker News API error:", response.status_code, response.text)
        return []

    ids = response.json()[:5]
    stories = []
    for story_id in ids:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item_resp = requests.get(item_url)
        if item_resp.status_code == 200:
            item = item_resp.json()
            title = item.get("title", "(no title)")
            link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
            stories.append(format_doc(title, "Hacker News", link))
    return stories

# ---------------------------
# Reddit Fetcher (r/devops)
# ---------------------------
def fetch_latest_reddit():
    headers = {"User-Agent": "AI-Meeting-Assistant/0.1 by openai"}
    url = "https://www.reddit.com/r/devops/hot.json?limit=5"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Reddit API error:", response.status_code, response.text)
        return []

    data = response.json()
    posts = []
    for post in data.get("data", {}).get("children", []):
        title = post["data"].get("title")
        link = "https://reddit.com" + post["data"].get("permalink")
        posts.append(format_doc(title, "Reddit", link))
    return posts


# ---------------------------
# Dev.to Fetcher
# ---------------------------
def fetch_latest_devto():
    url = "https://dev.to/api/articles?tag=softwaredevelopment&top=5"
    response = requests.get(url)
    if response.status_code != 200:
        print("Dev.to API error:", response.status_code, response.text)
        return []

    data = response.json()
    articles = []
    for article in data[:5]:
        articles.append(format_doc(article.get("title"), "Dev.to", article.get("url")))
    return articles

# ---------------------------
# DZone Fetcher
# ---------------------------
def fetch_latest_dzone():
    url = "https://dzone.com/rss"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    posts = []
    for item in soup.find_all("item")[:5]:
        title = item.title.text
        link = item.link.text
        posts.append(format_doc(title, "DZone", link))
    return posts

# ---------------------------
# InfoQ Fetcher
# ---------------------------
def fetch_latest_infoq():
    url = "https://feed.infoq.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    posts = []
    for item in soup.find_all("item")[:5]:
        title = item.title.text
        link = item.link.text
        posts.append(format_doc(title, "InfoQ", link))
    return posts


# ---------------------------
# Medium Fetcher (limited via RSS)
# ---------------------------
def fetch_latest_medium():
    url = "https://medium.com/feed/tag/software-development"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    posts = []
    for item in soup.find_all("item")[:5]:
        title = item.title.text
        link = item.link.text
        posts.append(format_doc(title, "Medium", link))
    return posts

def fetch_github_discussions():
    discussions = []
    trending_repos = [
        "vercel/next.js",
        "microsoft/vscode",
        "facebook/react",
        "torvalds/linux"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html",
    }

    for repo in trending_repos:
        url = f"https://github.com/{repo}/discussions"
        try:
            resp = requests.get(url, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            titles = soup.select("a.Link--primary.v-align-middle.no-underline.h4")[:3]
            for t in titles:
                title = t.text.strip()
                link = "https://github.com" + t["href"]
                discussions.append(format_doc(title, f"{repo} Discussion", link))
        except Exception as e:
            print(f"Error fetching discussions from {repo}:", e)

    return discussions
def fetch_scholar_papers():
    papers = []
    try:
        query = scholarly.search_pubs("software engineering best practices")
        for i in range(5):
            paper = next(query)
            title = paper.get("bib", {}).get("title", "Untitled")
            link = paper.get("pub_url") or f"https://scholar.google.com/scholar?q={title.replace(' ', '+')}"
            papers.append(format_doc(title, "Google Scholar", link))
    except Exception as e:
        print("Error fetching Google Scholar papers:", e)
    return papers
# ---------------------------
# Aggregator
# ---------------------------
def fetch_all_sources():
    all_results = []
    for fetcher in [
        fetch_latest_github,
        fetch_latest_stackoverflow,
        fetch_latest_arxiv,
        fetch_latest_rfc,
        fetch_latest_hackernews,
        fetch_latest_reddit,
        fetch_latest_devto,
        fetch_latest_dzone,
        fetch_latest_infoq,
        fetch_latest_medium,
        fetch_github_discussions,
        fetch_scholar_papers
    ]:
        try:
            all_results += fetcher()
        except Exception as e:
            print(f"Error in {fetcher.__name__}:", e)
    return all_results

# ---------------------------
# Save results for debugging or reuse
# ---------------------------
def save_to_file(data, path="fetched_docs.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Main Runner
# ---------------------------
# if __name__ == "__main__":
#     docs = fetch_all_sources()
#     for doc in docs:
#         print(f"{doc['source']}: {doc['text']} -> {doc['link']}")
#     save_to_file(docs)



