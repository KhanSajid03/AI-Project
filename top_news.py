from __future__ import annotations

import argparse
import email.utils
import html
import math
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit


DEFAULT_FEEDS = (
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://www.france24.com/en/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
)
MAX_FEED_BYTES = 5 * 1024 * 1024
STOP_WORDS = {
    "about", "after", "amid", "and", "are", "around", "been", "before",
    "being", "between", "breaking", "could", "during", "from", "have",
    "into", "just", "more", "most", "over", "says", "said", "say",
    "that", "their", "there", "these", "they", "this", "those", "today",
    "told", "too", "under", "update", "updated", "video", "watch", "week",
    "were", "what", "when", "where", "which", "while", "with", "would",
    "year", "years", "your",
}
WEAK_MATCH_WORDS = {
    "attack", "attacks", "dead", "gunmen", "injured", "kill", "killed",
    "protest", "protests", "shooting", "shot", "student", "students",
    "wounded",
}


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    source: str
    published: datetime


@dataclass
class Topic:
    articles: list[Article] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {article.source for article in self.articles}

    @property
    def latest(self) -> datetime:
        return max(article.published for article in self.articles)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def child_text(element: ET.Element, *names: str) -> str | None:
    wanted = {name.lower() for name in names}
    for child in element:
        if local_name(child.tag) in wanted:
            text = "".join(child.itertext()).strip()
            if text:
                return html.unescape(text)
    return None


def parse_date(value: str) -> datetime | None:
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_feed(data: bytes, feed_url: str) -> tuple[str, list[Article]]:
    root = ET.fromstring(data)
    channel = next(
        (element for element in root.iter() if local_name(element.tag) == "channel"),
        root,
    )
    source = child_text(channel, "title") or urlsplit(feed_url).hostname or feed_url
    articles: list[Article] = []

    for item in root.iter():
        if local_name(item.tag) not in {"item", "entry"}:
            continue
        title = child_text(item, "title")
        url = child_text(item, "link")
        if not url:
            link_element = next(
                (
                    child for child in item
                    if local_name(child.tag) == "link" and child.get("href")
                ),
                None,
            )
            if link_element is not None:
                url = link_element.get("href")
        date_text = child_text(item, "pubdate", "published", "updated", "date")
        published = parse_date(date_text) if date_text else None
        if title and url and published:
            articles.append(
                Article(
                    title=html.unescape(title),
                    url=url.strip(),
                    source=source,
                    published=published,
                )
            )
    return source, articles


def fetch_feed(feed_url: str) -> tuple[str, list[Article]]:
    request = urllib.request.Request(
        feed_url,
        headers={"User-Agent": "weekly-news-topics/1.0 (+RSS reader)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(MAX_FEED_BYTES + 1)
    if len(data) > MAX_FEED_BYTES:
        raise ValueError(f"feed exceeds the {MAX_FEED_BYTES}-byte size limit")
    return parse_feed(data, feed_url)


def normalized_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def title_tokens(title: str) -> set[str]:
    normalized = title.lower().replace("u.s.", "united states").replace("u.k.", "united kingdom")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3
        and token not in STOP_WORDS
        and token not in WEAK_MATCH_WORDS
    }


def cosine_similarity(
    left: set[str], right: set[str], inverse_frequency: dict[str, float]
) -> float:
    shared = left & right
    if not shared:
        return 0.0
    left_norm = math.sqrt(sum(inverse_frequency[token] ** 2 for token in left))
    right_norm = math.sqrt(sum(inverse_frequency[token] ** 2 for token in right))
    if not left_norm or not right_norm:
        return 0.0
    dot_product = sum(inverse_frequency[token] ** 2 for token in shared)
    return dot_product / (left_norm * right_norm)


def cluster_articles(articles: Sequence[Article]) -> list[Topic]:
    token_sets = [title_tokens(article.title) for article in articles]
    document_frequency: Counter[str] = Counter()
    for tokens in token_sets:
        document_frequency.update(tokens)
    count = len(articles)
    inverse_frequency = {
        token: math.log((count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }

    topics: list[Topic] = []
    ordered = sorted(
        zip(articles, token_sets),
        key=lambda pair: pair[0].published,
        reverse=True,
    )
    for article, tokens in ordered:
        if not tokens:
            continue
        best_topic: Topic | None = None
        best_similarity = 0.0
        for topic in topics:
            similarities = []
            for member in topic.articles:
                member_tokens = title_tokens(member.title)
                if len(tokens & member_tokens) >= 2:
                    similarities.append(
                        cosine_similarity(tokens, member_tokens, inverse_frequency)
                    )
            similarity = max(similarities, default=0.0)
            if similarity > best_similarity:
                best_similarity = similarity
                best_topic = topic
        if best_topic is not None and best_similarity >= 0.20:
            best_topic.articles.append(article)
        else:
            topics.append(Topic(articles=[article]))

    for topic in topics:
        unique_articles: dict[str, Article] = {}
        for article in topic.articles:
            key = normalized_url(article.url) or article.title.lower()
            unique_articles.setdefault(key, article)
        topic.articles = list(unique_articles.values())

    return sorted(
        topics,
        key=lambda topic: (len(topic.sources), len(topic.articles), topic.latest),
        reverse=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Find five frequently covered news topics from publisher RSS feeds "
            "published within the last seven days."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="rolling number of days to include (default: 7)",
    )
    parser.add_argument(
        "--feed",
        action="append",
        dest="feeds",
        help="RSS or Atom feed URL; may be repeated to replace the defaults",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=args.days)
    feed_urls = args.feeds or list(DEFAULT_FEEDS)
    articles: list[Article] = []
    successful_feeds = 0

    for feed_url in feed_urls:
        try:
            source, feed_articles = fetch_feed(feed_url)
        except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError, ValueError) as exc:
            print(f"Could not read feed {feed_url}: {exc}", file=sys.stderr)
            continue
        successful_feeds += 1
        articles.extend(
            article
            for article in feed_articles
            if cutoff <= article.published <= now
        )
        print(
            f"Read {len(feed_articles)} dated headlines from {source}",
            file=sys.stderr,
        )

    if successful_feeds == 0:
        print("No feeds could be read; check the feed URLs and your connection.", file=sys.stderr)
        return 1
    if not articles:
        print(f"No dated headlines were found in the last {args.days} days.")
        return 0

    topics = cluster_articles(articles)[:5]
    print(f"Top news topics from the last {args.days} days (UTC)")
    print(
        "Ranked by the number of distinct feeds covering a topic, then "
        "headline count. Results reflect only the selected feeds.\n"
    )
    for index, topic in enumerate(topics, start=1):
        representative = max(topic.articles, key=lambda item: item.published)
        print(
            f"{index}. {representative.title} "
            f"({len(topic.sources)} source(s), {len(topic.articles)} headline(s))"
        )
        for article in sorted(
            topic.articles,
            key=lambda item: (item.source, item.published),
            reverse=True,
        )[:5]:
            print(f"   - {article.source}: {article.title}")
            print(f"     {article.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
