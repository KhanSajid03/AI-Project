import unittest
from datetime import datetime, timezone

from top_news import Article, cluster_articles, parse_feed


class ParseFeedTests(unittest.TestCase):
    def test_parses_rss_and_channel_source(self) -> None:
        xml = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel><title>Example News</title>
          <item>
            <title>Example event reported today</title>
            <link>https://example.com/story</link>
            <pubDate>Fri, 23 Oct 2026 10:00:00 GMT</pubDate>
          </item>
        </channel></rss>"""

        source, articles = parse_feed(xml, "https://example.com/rss")

        self.assertEqual(source, "Example News")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://example.com/story")
        self.assertEqual(
            articles[0].published,
            datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc),
        )

    def test_parses_rdf_atom_style_feed_links(self) -> None:
        xml = b"""<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns="http://purl.org/rss/1.0/">
          <channel><title>Example Wire</title></channel>
          <item><title>Example event update</title>
            <link>https://example.com/event</link>
            <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">
              2026-10-23T10:00:00Z
            </dc:date>
          </item>
        </rdf:RDF>"""

        source, articles = parse_feed(xml, "https://example.com/feed.rdf")

        self.assertEqual(source, "Example Wire")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://example.com/event")


class ClusterArticlesTests(unittest.TestCase):
    def test_groups_matching_coverage_and_ranks_multi_source_topics_first(self) -> None:
        published = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)
        articles = [
            Article(
                "Ebola vaccine trial begins in Congo",
                "https://one.example/ebola",
                "One News",
                published,
            ),
            Article(
                "Congo begins Ebola vaccine trial",
                "https://two.example/ebola",
                "Two News",
                published,
            ),
            Article(
                "Local football club wins weekend match",
                "https://one.example/football",
                "One News",
                published,
            ),
        ]

        topics = cluster_articles(articles)

        self.assertEqual(len(topics), 2)
        self.assertEqual(topics[0].sources, {"One News", "Two News"})
        self.assertEqual(len(topics[0].articles), 2)
        self.assertEqual(topics[1].sources, {"One News"})

    def test_groups_differently_worded_coverage_of_the_same_event(self) -> None:
        published = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)
        headlines = [
            ("Seven shot dead by gunmen at church funeral in Jordan", "One News"),
            ("Masked gunmen kill seven in attack on church funeral in Jordan", "Two News"),
            ("Jordan: Karak church shooting leaves 7 dead, dozens injured", "Three News"),
            ("Several killed, dozens wounded in Jordan church shooting", "Four News"),
            (
                "Israeli forces begin demolishing Palestinian buildings in Hebron after soldiers shot dead",
                "Five News",
            ),
            (
                "French student protests continue after meeting with Macron",
                "Six News",
            ),
        ]
        articles = [
            Article(title, f"https://{source.lower().replace(' ', '')}.example/{index}", source, published)
            for index, (title, source) in enumerate(headlines)
        ]

        topics = cluster_articles(articles)

        self.assertEqual(len(topics), 3)
        jordan_topic = next(topic for topic in topics if "One News" in topic.sources)
        self.assertEqual(
            jordan_topic.sources,
            {"One News", "Two News", "Three News", "Four News"},
        )


if __name__ == "__main__":
    unittest.main()
