import hashlib
import html
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import feedparser
import requests
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
DB_PATH = "portal_cache.db"
USER_AGENT = "VictorConsultingPolicyPortalMVP/0.1 (+https://victorconsulting.jp/)"
TIMEOUT = 60
MAX_BODY_CHARS = 1800
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    kind: str  # rss | html
    url: str
    enabled: bool = True
    note: str = ""


SOURCES = [
    Source(
        key="meti",
        name="経済産業省",
        kind="rss",
        url="https://www.meti.go.jp/ml_index_release_atom.xml",
        enabled=True,
    ),
    Source(
        key="chusho",
        name="中小企業庁",
        kind="rss",
        url="https://www.chusho.meti.go.jp/rss/index.xml",
        enabled=True,
    ),
    Source(
        key="nta",
        name="国税庁",
        kind="html",
        url="https://www.nta.go.jp/merumaga/",
        enabled=True,
        note="MVPではメール配信案内ページから新着導線を取得。将来は別の新着一覧へ差し替え推奨。",
    ),
    Source(
        key="fsa",
        name="金融庁",
        kind="rss",
        url="https://www.fsa.go.jp/fsaNewsListAll_rss2.xml",
        enabled=True,
    ),
    Source(
        key="mhlw",
        name="厚生労働省",
        kind="rss",
        url="https://www.mhlw.go.jp/rss/new.rdf",
        enabled=False,
        note="厚生労働省RSSページには再配布を断る注意書きあり。運用開始前に法務・利用条件確認のうえ有効化。",
    ),
]

CATEGORY_RULES = {
    "補助金・助成金": ["補助金", "助成金", "公募", "交付", "支援事業"],
    "税制・会計": ["税", "申告", "納税", "インボイス", "電子帳簿", "源泉"],
    "雇用・労務": ["雇用", "労働", "賃金", "育児", "介護", "年金"],
    "金融・資金繰り": ["金融", "保証", "融資", "資金繰り", "貸付"],
    "制度改正": ["改正", "施行", "見直し", "改定"],
    "調査・統計": ["調査", "統計", "白書", "レポート", "結果"],
    "注意喚起": ["注意", "注意喚起", "不審", "詐欺", "警告"],
}


# ---------- DB ----------
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL,
                source_name TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                published_at TEXT,
                body_text TEXT,
                summary_short TEXT,
                summary_three TEXT,
                summary_business TEXT,
                category TEXT,
                importance_score INTEGER DEFAULT 0,
                importance_label TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
        if "importance_score" not in existing_columns:
            conn.execute("ALTER TABLE articles ADD COLUMN importance_score INTEGER DEFAULT 0")
        if "importance_label" not in existing_columns:
            conn.execute("ALTER TABLE articles ADD COLUMN importance_label TEXT DEFAULT ''")
        conn.commit()


def upsert_article(article: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO articles (
                source_key, source_name, title, url, published_at, body_text,
                summary_short, summary_three, summary_business, category,
                importance_score, importance_label,
                created_at, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                published_at=excluded.published_at,
                body_text=excluded.body_text,
                summary_short=excluded.summary_short,
                summary_three=excluded.summary_three,
                summary_business=excluded.summary_business,
                category=excluded.category,
                importance_score=excluded.importance_score,
                importance_label=excluded.importance_label,
                content_hash=excluded.content_hash
            """,
            (
                article["source_key"],
                article["source_name"],
                article["title"],
                article["url"],
                article["published_at"],
                article["body_text"],
                article["summary_short"],
                article["summary_three"],
                article["summary_business"],
                article["category"],
                article["importance_score"],
                article["importance_label"],
                article["created_at"],
                article["content_hash"],
            ),
        )
        conn.commit()


def load_articles(limit: int = 200) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM articles
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Fetch helpers ----------
def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def parse_datetime(entry) -> Optional[str]:
    for attr in ["published_parsed", "updated_parsed"]:
        value = getattr(entry, attr, None)
        if value:
            return datetime(*value[:6], tzinfo=timezone.utc).astimezone(JST).isoformat()
    return None


def infer_category(title: str, body: str) -> str:
    text = f"{title} {body}"
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "その他"


def summarize_text_rule_based(title: str, body: str) -> tuple[str, str, str]:
    body = clean_text(body)
    if not body:
        short = title
        three = "本文取得前のため、原文リンクで内容をご確認ください。"
        business = "実務への影響は原文確認後に判断してください。"
        return short, three, business

    sentence_candidates = re.split(r"(?<=[。！？])", body)
    sentence_candidates = [s.strip() for s in sentence_candidates if s.strip()]
    first = sentence_candidates[0] if sentence_candidates else body[:120]
    second = sentence_candidates[1] if len(sentence_candidates) > 1 else "対象や詳細条件は原文をご確認ください。"
    third = sentence_candidates[2] if len(sentence_candidates) > 2 else "期限・要件・対象範囲の確認が重要です。"

    short = clean_text(first)[:90]
    three = "\n".join([f"・{clean_text(x)[:120]}" for x in [first, second, third]])
    business = (
        f"{title}に関する公表です。"
        f" 自社への影響があるか、対象者・開始時期・締切・必要手続を原文で確認してください。"
    )
    return short, three, business


def summarize_text_ai(title: str, body: str) -> Optional[tuple[str, str, str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
あなたは官公庁の新着情報を中小企業経営者向けに要約する編集者です。
以下の形式でJSONだけを返してください。
{{
  "summary_short": "90字以内の一行要約",
  "summary_three": "箇条書き3行。各行の先頭は・",
  "summary_business": "実務上の意味を120字以内で簡潔に"
}}

タイトル:
{title}

本文:
{body[:2500]}
""".strip()
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        text = response.output_text.strip()
        import json
        data = json.loads(text)
        return (
            clean_text(data.get("summary_short", ""))[:90],
            data.get("summary_three", ""),
            clean_text(data.get("summary_business", ""))[:140],
        )
    except Exception:
        return None


def summarize_text(title: str, body: str) -> tuple[str, str, str]:
    ai_result = summarize_text_ai(title, body)
    if ai_result:
        return ai_result
    return summarize_text_rule_based(title, body)


def calculate_importance(title: str, body: str, category: str) -> tuple[int, str]:
    text = f"{title} {body}"
    score = 1
    high_keywords = ["公募", "募集", "開始", "締切", "交付決定", "改正", "施行", "延長", "注意喚起"]
    medium_keywords = ["補助金", "助成金", "税", "金融", "賃金", "支援", "制度"]

    if category in {"補助金・助成金", "税制・会計", "制度改正", "金融・資金繰り", "注意喚起"}:
        score += 1
    score += sum(1 for kw in high_keywords if kw in text)
    score += min(2, sum(1 for kw in medium_keywords if kw in text))

    score = max(1, min(score, 5))
    label_map = {
        5: "★★★★★ 非常に重要",
        4: "★★★★☆ 重要",
        3: "★★★☆☆ 注目",
        2: "★★☆☆☆ 参考",
        1: "★☆☆☆☆ 低め",
    }
    return score, label_map[score]


PROFILE_PRESETS = {
    "製造業 × 中小企業": {
        "include_any": ["中小企業", "中堅", "製造", "工場", "設備", "省エネ", "GX", "補助金", "税制", "生産性"],
        "boost_any": ["補助金", "設備投資", "公募", "省エネ", "賃上げ", "価格転嫁", "資金繰り"],
        "exclude_any": ["個人向け", "年金受給者", "医療保険"],
    },
    "税務・会計重視": {
        "include_any": ["税", "申告", "納税", "インボイス", "電子帳簿", "会計", "控除"],
        "boost_any": ["期限", "改正", "開始", "延長"],
        "exclude_any": [],
    },
    "雇用・労務重視": {
        "include_any": ["雇用", "労働", "賃金", "最低賃金", "育児", "介護", "助成金"],
        "boost_any": ["改正", "施行", "募集", "申請"],
        "exclude_any": [],
    },
    "金融・資金繰り重視": {
        "include_any": ["金融", "保証", "融資", "資金繰り", "貸付", "返済", "条件変更"],
        "boost_any": ["制度", "延長", "開始", "支援"],
        "exclude_any": [],
    },
}


def score_profile_match(article: dict, preset_name: str) -> tuple[int, str]:
    preset = PROFILE_PRESETS.get(preset_name)
    if not preset:
        return 0, ""

    text = f"{article.get('title', '')} {article.get('body_text', '')} {article.get('category', '')}"
    include_hits = sum(1 for kw in preset["include_any"] if kw in text)
    boost_hits = sum(1 for kw in preset["boost_any"] if kw in text)
    exclude_hits = sum(1 for kw in preset["exclude_any"] if kw in text)

    raw_score = include_hits * 2 + boost_hits - exclude_hits * 3
    if raw_score >= 8:
        return 5, "★★★★★ 高適合"
    if raw_score >= 5:
        return 4, "★★★★☆ 適合"
    if raw_score >= 3:
        return 3, "★★★☆☆ 一部適合"
    if raw_score >= 1:
        return 2, "★★☆☆☆ 低め"
    return 1, "★☆☆☆☆ ほぼ対象外"


def response_to_text(response: requests.Response) -> str:
    if not response.encoding or response.encoding.lower() in {"iso-8859-1", "latin-1"}:
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def extract_article_body(url: str, session: requests.Session) -> str:
    try:
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
    except Exception:
        return ""
    text = html_to_text(response_to_text(response))
    return text[:MAX_BODY_CHARS]


# ---------- Source adapters ----------
def fetch_from_rss(source: Source, session: requests.Session, max_items: int = 20) -> list[dict]:
    response = session.get(source.url, timeout=TIMEOUT)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    items: list[dict] = []
    for entry in feed.entries[:max_items]:
        url = getattr(entry, "link", "")
        title = clean_text(getattr(entry, "title", "無題"))
        summary = clean_text(getattr(entry, "summary", ""))
        body = summary
        if len(body) < 120 and url:
            body = extract_article_body(url, session)
        category = infer_category(title, body)
        summary_short, summary_three, summary_business = summarize_text(title, body)
        importance_score, importance_label = calculate_importance(title, body, category)
        content_hash = hashlib.sha256(f"{title}|{url}|{body}".encode("utf-8")).hexdigest()
        items.append(
            {
                "source_key": source.key,
                "source_name": source.name,
                "title": title,
                "url": url,
                "published_at": parse_datetime(entry),
                "body_text": body,
                "summary_short": summary_short,
                "summary_three": summary_three,
                "summary_business": summary_business,
                "category": category,
                "importance_score": importance_score,
                "importance_label": importance_label,
                "created_at": datetime.now(JST).isoformat(),
                "content_hash": content_hash,
            }
        )
    return items


def fetch_from_nta_html(source: Source, session: requests.Session, max_items: int = 20) -> list[dict]:
    response = session.get(source.url, timeout=TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response_to_text(response), "html.parser")

    items: list[dict] = []
    links = soup.select("a[href]")
    seen = set()
    for a in links:
        href = a.get("href", "")
        text = clean_text(a.get_text(" "))
        if not href or not text:
            continue
        if "backnumber" not in href and "news_toroku" not in href and "merumaga" not in href:
            continue
        if href.startswith("/"):
            url = f"https://www.nta.go.jp{href}"
        elif href.startswith("http"):
            url = href
        else:
            url = f"https://www.nta.go.jp/merumaga/{href.lstrip('./')}"
        if url in seen:
            continue
        seen.add(url)
        body = extract_article_body(url, session)
        category = infer_category(text, body)
        summary_short, summary_three, summary_business = summarize_text(text, body)
        importance_score, importance_label = calculate_importance(text, body, category)
        content_hash = hashlib.sha256(f"{text}|{url}|{body}".encode("utf-8")).hexdigest()
        items.append(
            {
                "source_key": source.key,
                "source_name": source.name,
                "title": text,
                "url": url,
                "published_at": None,
                "body_text": body,
                "summary_short": summary_short,
                "summary_three": summary_three,
                "summary_business": summary_business,
                "category": category,
                "importance_score": importance_score,
                "importance_label": importance_label,
                "created_at": datetime.now(JST).isoformat(),
                "content_hash": content_hash,
            }
        )
        if len(items) >= max_items:
            break
    return items


def refresh_sources(selected_sources: Iterable[str]) -> tuple[int, list[str]]:
    session = get_session()
    inserted = 0
    notes: list[str] = []
    processed_sources = 0
    for source in SOURCES:
        if source.key not in selected_sources:
            continue
        if not source.enabled:
            notes.append(f"{source.name}: 現在は無効です。{source.note}")
            continue
        try:
            if source.kind == "rss":
                items = fetch_from_rss(source, session)
            elif source.key == "nta":
                items = fetch_from_nta_html(source, session)
            else:
                items = []
            processed_sources += 1
            notes.append(f"{source.name}: {len(items)}件取得")
            for item in items:
                upsert_article(item)
                inserted += 1
        except Exception as exc:
            notes.append(f"{source.name}: 取得失敗 - {exc}")
        if source.note:
            notes.append(f"{source.name}: {source.note}")
    if processed_sources == 0:
        notes.append("有効な取得対象が選択されていません。")
    return inserted, notes


# ---------- UI ----------
st.set_page_config(page_title="政策ポータル MVP", layout="wide")
init_db()

st.title("政策ポータル MVP")
st.caption("経済産業省・中小企業庁・国税庁・金融庁などの新着を横断し、概要だけを読むための試作版")

with st.sidebar:
    st.header("更新")
    ai_ready = bool(os.getenv("OPENAI_API_KEY", "").strip()) and OpenAI is not None
    st.write(f"要約方式: {'AI要約' if ai_ready else 'ルールベース要約'}")
    profile_preset = st.selectbox("自分向けフィルター", options=["使わない"] + list(PROFILE_PRESETS.keys()))
    default_keys = [s.key for s in SOURCES if s.enabled]
    selected = st.multiselect(
        "取得対象",
        options=[s.key for s in SOURCES],
        default=default_keys,
        format_func=lambda x: next(s.name for s in SOURCES if s.key == x),
    )
    if st.button("最新情報を取得", use_container_width=True):
        with st.spinner("取得中..."):
            inserted, notes = refresh_sources(selected)
        st.success(f"更新完了: {inserted}件を処理しました。")
        for note in notes:
            st.info(note)
        refreshed_count = len(load_articles(300))
        st.caption(f"DB保存後の記事件数: {refreshed_count}件")

    st.markdown("---")
    st.markdown("**運用メモ**")
    st.write("- AI要約は現在ルールベース")
    st.write("- 各記事で必ず原文リンクを確認")
    st.write("- 免責文を公開時に明記")

articles = load_articles(300)

st.subheader("今日の重要政策")
today_key = datetime.now(JST).date().isoformat()
important_today = [
    a for a in articles
    if (a.get("published_at") or a.get("created_at", "")).startswith(today_key)
]
important_today = sorted(important_today, key=lambda x: (x.get("importance_score") or 0), reverse=True)[:5]
if important_today:
    for a in important_today:
        st.markdown(f"- **{a['source_name']}** | {a['importance_label']} | {a['title']}")
else:
    st.caption("本日公開分がまだ少ないため、重要政策表示はこれから蓄積されます。")

if not articles:
    st.info("まだデータがありません。左側の『最新情報を取得』を押してください。")
    st.stop()

# Filters
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    source_filter = st.selectbox(
        "省庁",
        options=["すべて"] + sorted({a["source_name"] for a in articles}),
    )
with col2:
    category_filter = st.selectbox(
        "カテゴリ",
        options=["すべて"] + sorted({a["category"] for a in articles}),
    )
with col3:
    keyword = st.text_input("キーワード検索", placeholder="例：補助金 / 税制 / 賃上げ")

filtered = articles
if profile_preset != "使わない":
    for a in filtered:
        match_score, match_label = score_profile_match(a, profile_preset)
        a["profile_match_score"] = match_score
        a["profile_match_label"] = match_label
    filtered = [a for a in filtered if a.get("profile_match_score", 0) >= 3]
    filtered = sorted(filtered, key=lambda x: (x.get("profile_match_score", 0), x.get("importance_score", 0)), reverse=True)
if source_filter != "すべて":
    filtered = [a for a in filtered if a["source_name"] == source_filter]
if category_filter != "すべて":
    filtered = [a for a in filtered if a["category"] == category_filter]
if keyword:
    filtered = [
        a
        for a in filtered
        if keyword.lower() in (a["title"] + " " + (a["body_text"] or "")).lower()
    ]

if profile_preset != "使わない":
    st.subheader(f"新着一覧 ({len(filtered)}件) - {profile_preset}向け")
    top_matches = filtered[:5]
    if top_matches:
        st.caption("自分向け上位候補")
        for a in top_matches:
            st.markdown(f"- **{a['profile_match_label']}** | {a['source_name']} | {a['title']}")
else:
    st.subheader(f"新着一覧 ({len(filtered)}件)")

for article in filtered:
    with st.container(border=True):
        if profile_preset != "使わない" and "profile_match_score" not in article:
            article["profile_match_score"], article["profile_match_label"] = score_profile_match(article, profile_preset)
        widths = [1, 1, 2, 2, 2] if profile_preset != "使わない" else [1, 1, 2, 3]
        meta_cols = st.columns(widths)
        meta_cols[0].markdown(f"**{article['source_name']}**")
        meta_cols[1].markdown(f"`{article['category']}`")
        published_label = article["published_at"][:10] if article["published_at"] else "日時不明"
        meta_cols[2].markdown(f"{published_label}")
        meta_cols[3].markdown(article.get("importance_label", ""))
        if profile_preset != "使わない":
            meta_cols[4].markdown(article.get("profile_match_label", ""))

        st.markdown(f"### {article['title']}")
        st.write(article["summary_short"])

        with st.expander("3行要約と実務メモ"):
            st.text(article["summary_three"])
            st.write(article["summary_business"])
            if article["body_text"]:
                st.caption(article["body_text"][:400] + ("..." if len(article["body_text"]) > 400 else ""))

        st.link_button("原文を確認", article["url"])

st.markdown("---")
st.caption(
    "免責: 本アプリは公的機関の公式サイトではありません。要約は試作版であり、必ず原文をご確認ください。"
)

