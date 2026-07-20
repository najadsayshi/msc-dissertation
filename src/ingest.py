# ingest.py - download a 10-K from SEC EDGAR, save it as plain text in data/

import os
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


# Find the URL of the FY2023 10-K. Can't just take the newest 10-K,
# that could be a different fiscal year.
def get_10k_url(cik: str, fiscal_year: str = FISCAL_YEAR) -> str:
    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]
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


def save_text(text: str, company_name: str) -> str:
    # e.g. data/apple_10k.txt
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{company_name.lower().replace(' ', '_')}_10k.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


def main():
    ticker = "AAPL"
    company = COMPANIES[ticker]

    print(f"Looking up 10-K filing for {company['name']}...")
    url = get_10k_url(company["cik"])
    print(f"Found: {url}")

    print("Downloading and parsing HTML")
    text = download_and_parse(url)

    filepath = save_text(text, company["name"])
    print(f"Done. Saved {len(text):,} characters to {filepath}")


if __name__ == "__main__":
    main()
