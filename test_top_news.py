import json
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from top_news import (
    Article,
    cluster_articles,
    extract_article_text,
    environ,
    load_env_file,
    main,
    parse_feed,
    summarize_article,
)


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


class ArticleSummaryTests(unittest.TestCase):
    def test_loads_env_file_without_overriding_existing_environment(self) -> None:
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# API configuration\n"
                "OPENAI_API_KEY='file-key'\n"
                "OPENAI_MODEL=gpt-test\n",
                encoding="utf-8",
            )
            with patch.dict(
                "top_news.environ",
                {"OPENAI_API_KEY": "existing-key"},
                clear=True,
            ):
                load_env_file(env_path)

                self.assertEqual(
                    environ["OPENAI_API_KEY"],
                    "existing-key",
                )
                self.assertEqual(
                    environ["OPENAI_MODEL"],
                    "gpt-test",
                )

    def test_extracts_article_paragraphs_without_navigation(self) -> None:
        document = """
        <html><body>
          <nav><p>Navigation text should not be included.</p></nav>
          <article>
            <h1>Article heading</h1>
            <p>This is the article body with enough text to be extracted.</p>
          </article>
          <script><p>Script text should not be included.</p></script>
        </body></html>
        """

        text = extract_article_text(document)

        self.assertIn("Article heading", text)
        self.assertIn("This is the article body", text)
        self.assertNotIn("Navigation text", text)
        self.assertNotIn("Script text", text)

    def test_sends_source_grounded_prompt_to_openai_responses_api(self) -> None:
        article = Article(
            "Example headline",
            "https://example.com/story",
            "Example News",
            datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc),
        )
        response = BytesIO(
            json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "A concise article summary.",
                                }
                            ],
                        }
                    ]
                }
            ).encode()
        )

        with patch("top_news.urllib.request.urlopen", return_value=response) as urlopen:
            summary = summarize_article(article, "Verified article text.", "test-key", "test-model")

        self.assertEqual(summary, "A concise article summary.")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        sent_body = json.loads(request.data)
        self.assertEqual(sent_body["model"], "test-model")
        self.assertIn("Verified article text.", sent_body["input"])

    def test_main_writes_five_source_linked_summaries(self) -> None:
        published = datetime.now(timezone.utc)
        headlines = [
            "Earthquake disrupts transport in Peru",
            "Central bank raises rates after inflation report",
            "New telescope discovers distant planet",
            "Wildfire prompts evacuation in coastal town",
            "National team wins international football final",
        ]
        articles = [
            Article(
                title,
                f"https://example.com/story-{index}",
                "Example News",
                published,
            )
            for index, title in enumerate(headlines)
        ]
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "summary.txt"
            output = StringIO()
            errors = StringIO()
            with (
                patch.dict(
                    "top_news.environ",
                    {"OPENAI_API_KEY": "test-key"},
                    clear=True,
                ),
                patch("top_news.fetch_feed", return_value=("Example News", articles)),
                patch(
                    "top_news.fetch_article_text",
                    return_value="Verified article content.",
                ),
                patch(
                    "top_news.summarize_article",
                    side_effect=lambda article, *_: f"Summary for {article.title}.",
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                result = main(["--output", str(output_path)])

            summary_file = output_path.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(summary_file.count("   Summary:"), 5)
        self.assertIn("https://example.com/story-", summary_file)
        self.assertIn("Summary for Earthquake disrupts transport in Peru.", summary_file)


if __name__ == "__main__":
    unittest.main()
