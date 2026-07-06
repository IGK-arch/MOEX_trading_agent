"""Реестр RSS-источников."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class FeedConfig:
    """Configuration for one news feed."""

    url: str
    name: str
    tier: str
    poll_interval: int
    language: str
    enabled: bool = True
    requires_proxy: bool = False
    insecure_ssl: bool = False
    notes: str = ""

TIER_S_FEEDS: list[FeedConfig] = [
    FeedConfig(
        url="https://iss.moex.com/iss/sitenews.json",
        name="MOEX ISS sitenews",
        tier="S",
        poll_interval=30,
        language="ru",
        enabled=False,
        notes="JSON, handled by moex_iss_parser.py (not generic RSS)",
    ),
    FeedConfig(
        url="https://www.e-disclosure.ru/api/rss/randomevents",
        name="E-disclosure events",
        tier="S",
        poll_interval=30,
        language="ru",
        enabled=False,
        notes="Phase 16 audit (v0.0.12) — feed returns malformed XML "
        "('mismatched tag :6:68') consistently. Disabled until they "
        "fix it. Tier S sanctions/halts coverage via OFAC+UK+EU and "
        "MOEX ISS sitenews still works.",
    ),
    FeedConfig(
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
        name="Federal Reserve monetary",
        tier="S",
        poll_interval=30,
        language="en",
    ),
    FeedConfig(
        url="https://www.ecb.europa.eu/rss/press.html",
        name="ECB press",
        tier="S",
        poll_interval=30,
        language="en",
    ),
    FeedConfig(
        url="https://home.treasury.gov/news/rss",
        name="US Treasury press",
        tier="S",
        poll_interval=60,
        language="en",
        enabled=False,
        notes="Phase 12 audit (2026-05) — all RSS paths confirmed dead. "
        "OFAC recent-actions HTML scrape in SANCTIONS_SOURCES is the replacement.",
    ),
    FeedConfig(
        url="https://www.cbr.ru/rss/eventrss",
        name="CBR events & comments",
        tier="S",
        poll_interval=60,
        language="ru",
        notes="Bank of Russia rate decisions, official commentary. Critical for "
        "MOEX rate-sensitive sectors (banks, utilities).",
    ),
    FeedConfig(
        url="https://www.cbr.ru/rss/RssNews",
        name="CBR site news",
        tier="S",
        poll_interval=120,
        language="ru",
        notes="CBR general site updates (regulation, licensing, AML).",
    ),
]

TIER_A_FEEDS: list[FeedConfig] = [
    FeedConfig(
        url="https://www.interfax.ru/rss.asp",
        name="Interfax",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://tass.ru/rss/v2.xml",
        name="TASS economy",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://ria.ru/export/rss2/index.xml",
        name="RIA news",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://1prime.ru/export/rss2/index.xml",
        name="Prime (RIA financial wire)",
        tier="A",
        poll_interval=60,
        language="ru",
        notes="RIA Novosti's financial wire — heavy MOEX/issuer coverage.",
    ),
    FeedConfig(
        url="https://www.vedomosti.ru/rss/rubric/finance",
        name="Vedomosti finance",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://www.kommersant.ru/RSS/section-economics.xml",
        name="Kommersant economics",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://www.kommersant.ru/RSS/section-business.xml",
        name="Kommersant business",
        tier="A",
        poll_interval=120,
        language="ru",
        notes="Complements section-economics with corporate-level news.",
    ),
    FeedConfig(
        url="https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
        name="RBC news",
        tier="A",
        poll_interval=60,
        language="ru",
    ),
    FeedConfig(
        url="https://feeds.bloomberg.com/markets/news.rss",
        name="Bloomberg Markets",
        tier="A",
        poll_interval=60,
        language="en",
    ),
    FeedConfig(
        url="https://www.investing.com/rss/news.rss",
        name="Investing.com news",
        tier="A",
        poll_interval=60,
        language="en",
    ),
    FeedConfig(
        url="https://www.cnbc.com/id/19836768/device/rss/rss.html",
        name="CNBC Commodities",
        tier="A",
        poll_interval=60,
        language="en",
        enabled=False,
        notes="Phase 12 audit (2026-05) — CNBC retired /id/N/device/rss/rss.html "
        "pattern in 2026, all numeric IDs return 404. No public replacement.",
    ),
    FeedConfig(
        url="https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        name="WSJ Markets",
        tier="A",
        poll_interval=60,
        language="en",
    ),
    FeedConfig(
        url="https://www.themoscowtimes.com/rss/news",
        name="Moscow Times",
        tier="A",
        poll_interval=120,
        language="en",
        insecure_ssl=True,
        notes="Phase 16 audit (v0.0.12) — SSL cert verify intermittently fails "
        "from Yandex Cloud egress (self-signed chain). Use insecure_ssl "
        "since the source itself is well-known.",
    ),
    FeedConfig(
        url="https://www.intellinews.com/feed",
        name="bne IntelliNews CEE/CIS",
        tier="A",
        poll_interval=120,
        language="en",
        enabled=False,
        notes="Phase 12 audit (2026-05) — 403 for every UA. Cloudflare bot fight enabled.",
    ),
]

TIER_B_FEEDS: list[FeedConfig] = [
    FeedConfig(
        url="https://bcs-express.ru/rss",
        name="BCS Express",
        tier="B",
        poll_interval=300,
        language="ru",
        enabled=False,
        notes="Phase 16 audit (v0.0.12) — returns malformed XML "
        "('mismatched tag :6:68'). Either site is broken or "
        "Cloudflare anti-bot is injecting JS into XML response. "
        "Disabled. Smart-Lab + Finam analysis cover same Tier B.",
    ),
    FeedConfig(
        url="https://www.finam.ru/analysis/conews/rsspoint/",
        name="Finam analysis",
        tier="B",
        poll_interval=300,
        language="ru",
    ),
    FeedConfig(
        url="https://smart-lab.ru/rss/",
        name="Smart-Lab",
        tier="B",
        poll_interval=300,
        language="ru",
    ),
    FeedConfig(
        url="https://mfd.ru/news/rss/",
        name="MFD",
        tier="B",
        poll_interval=300,
        language="ru",
    ),
    FeedConfig(
        url="https://www.banki.ru/xml/news.rss",
        name="Banki.ru",
        tier="B",
        poll_interval=300,
        language="ru",
        enabled=False,
        notes="Phase 12 audit (2026-05) — 302 redirect loop (geo-gated, serves only "
        "from RU egress IPs). Mark requires_proxy if/when we add RU residential proxy.",
    ),
    FeedConfig(
        url="https://www.forbes.ru/newrss.xml",
        name="Forbes RU",
        tier="B",
        poll_interval=300,
        language="ru",
    ),
    FeedConfig(
        url="https://habr.com/ru/rss/hub/finance/all/?fl=ru",
        name="Habr finance hub",
        tier="B",
        poll_interval=600,
        language="ru",
        notes="RU tech-community finance hub — early signal for retail "
        "sentiment on broker/fintech moves.",
    ),
]

TIER_C_FEEDS: list[FeedConfig] = [
    FeedConfig(
        url="https://oilprice.com/rss/main",
        name="OilPrice all",
        tier="C",
        poll_interval=300,
        language="en",
        notes="Phase 12 audit (2026-05) — back online via /rss/main.",
    ),
    FeedConfig(
        url="https://www.eia.gov/rss/todayinenergy.xml",
        name="EIA Today in Energy",
        tier="C",
        poll_interval=900,
        language="en",
        enabled=False,
        notes="Phase 12 audit (2026-05) — persistent 503 from Yandex Cloud egress "
        "(EIA Akamai geo-rate-limits non-US IPs). Re-enable behind US proxy.",
    ),
    FeedConfig(
        url="https://www.kitco.com/rss/KitcoNews.xml",
        name="Kitco mining",
        tier="C",
        poll_interval=300,
        language="en",
        enabled=False,
        notes="Phase 12 audit — 404. Kitco changed feed structure; mining covered "
        "by OilPrice main only.",
    ),
    FeedConfig(
        url="https://www.mining.com/feed/",
        name="Mining.com",
        tier="C",
        poll_interval=300,
        language="en",
        enabled=False,
        notes="Phase 12 audit — 403 Forbidden (Cloudflare, no residential proxy).",
    ),
]

FEEDS: list[FeedConfig] = [
    *TIER_S_FEEDS,
    *TIER_A_FEEDS,
    *TIER_B_FEEDS,
    *TIER_C_FEEDS,
]

def feeds_by_tier(tier: str) -> list[FeedConfig]:
    """Feeds by tier."""
    return [f for f in FEEDS if f.tier == tier and f.enabled]

def feeds_by_language(language: str) -> list[FeedConfig]:
    """Feeds by language."""
    return [f for f in FEEDS if f.language == language and f.enabled]

def enabled_feeds() -> list[FeedConfig]:
    """Enabled feeds."""
    return [f for f in FEEDS if f.enabled]

SANCTIONS_SOURCES = {
    "ofac_sdn_xml": "https://www.treasury.gov/ofac/downloads/sdn.xml",
    "ofac_recent_actions": "https://ofac.treasury.gov/recent-actions",
    "eu_fsf_rss": "https://webgate.ec.europa.eu/fsd/fsf/public/rss",
    "uk_ofsi_atom": "https://www.gov.uk/government/organisations/office-of-financial-sanctions-implementation.atom",
}

EDISCLOSURE_URLS = {
    "events_list": "https://www.e-disclosure.ru/portal/event.aspx",
    "rss": "https://www.e-disclosure.ru/api/rss/randomevents",
}
