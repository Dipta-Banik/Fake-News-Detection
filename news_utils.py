import feedparser
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_articles(query):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:
        summary = entry.get("summary", "")
        articles.append({
            "title": entry.title,
            "summary": summary,
            "link": entry.link
        })
    return articles

def find_most_similar(user_text, articles):
    corpus = [user_text] + [a["title"] + " " + a["summary"] for a in articles]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    best_idx = similarities.argmax()
    return articles[best_idx], similarities[best_idx]

def extract_news_source(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.source_url or article.meta_data.get("og:site_name") or "Unknown Source"
    except:
        return "Unknown Source"