# ingest.py - download a 10-K from SEC EDGAR, save it as plain text in data/

import os
import argparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# SEC blocks anonymous requests - has to say who's asking
HEADERS = {
    "User-Agent": "Najad Marathumpalli mrnajadas@gmail.com"
}

# the five companies + their SEC CIK numbers
COMPANIES = {
    "AAPL": {"name": "Apple",             "cik": "320193"},
    "MSFT": {"name": "Microsoft",         "cik": "789019"},
    "JPM":  {"name": "JPMorgan Chase",    "cik": "19617"},
    "XOM":  {"name": "ExxonMobil",        "cik": "34088"},
    "JNJ":  {"name": "Johnson & Johnson", "cik": "200406"},
}

# match on the period the filing covers, not the filing date - fiscal years
# end in different months (Apple: September, Microsoft: June)
FISCAL_YEAR = "2023"


# scan one block of filings for the 10-K covering the fiscal year. Returns the
# URL, or None if this block doesn't have it
def find_10k_in_block(filings: dict, cik: str, fiscal_year: str):
    for i, form in enumerate(filings["form"]):
        report_date = filings["reportDate"][i]
        if form == "10-K" and report_date.startswith(fiscal_year):
            accession = filings["accessionNumber"][i]
            primary_doc = filings["primaryDocument"][i]
            accession_clean = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{accession_clean}/{primary_doc}"
            )
    return None


# Find the URL of the FY2023 10-K. Can't just take the newest 10-K,
# that could be a different fiscal year.
def get_10k_url(cik: str, fiscal_year: str = FISCAL_YEAR) -> str:
    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    found = find_10k_in_block(data["filings"]["recent"], cik, fiscal_year)
    if found:
        return found

    # "recent" only covers about the last year. Heavy filers like JPM push out
    # thousands of prospectuses, so their 10-K falls off into the older pages
    # listed under filings.files
    for page in data["filings"]["files"]:
        # nothing filed before the fiscal year started can be that year's 10-K
        if page["filingTo"] < fiscal_year:
            continue

        page_url = f"https://data.sec.gov/submissions/{page['name']}"
        page_response = requests.get(page_url, headers=HEADERS)
        page_response.raise_for_status()

        found = find_10k_in_block(page_response.json(), cik, fiscal_year)
        if found:
            return found

    raise ValueError(f"No FY{fiscal_year} 10-K filing found for CIK {cik}")


def download_and_parse(url: str) -> str:
    # grab the HTML, flatten it to plain text
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    # script/style tags are code, not report text
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


# where a ticker's text file lives. index.py imports this so both scripts
# agree on the name instead of each building its own
def text_path(ticker: str) -> str:
    # ticker not company name - "johnson_&_johnson_10k.txt" is a pain in a shell
    return f"data/{ticker.lower()}_10k.txt"


def save_text(text: str, ticker: str) -> str:
    # e.g. data/aapl_10k.txt
    os.makedirs("data", exist_ok=True)
    filepath = text_path(ticker)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


def fetch_company(ticker: str) -> None:
    company = COMPANIES[ticker]

    print(f"Looking up 10-K filing for {company['name']}...")
    url = get_10k_url(company["cik"])
    print(f"Found: {url}")

    print("Downloading and parsing HTML")
    text = download_and_parse(url)

    filepath = save_text(text, ticker)
    print(f"Done. Saved {len(text):,} characters to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Download a FY2023 10-K from SEC EDGAR.")
    parser.add_argument("--ticker", default="AAPL", choices=sorted(COMPANIES.keys()))
    parser.add_argument("--all", action="store_true", help="fetch all five companies")
    args = parser.parse_args()

    if args.all:
        tickers = sorted(COMPANIES.keys())
    else:
        tickers = [args.ticker]

    for ticker in tickers:
        fetch_company(ticker)


if __name__ == "__main__":
    main()
