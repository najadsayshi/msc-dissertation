import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# SEC EDGAR blocks requests without a descriptive User-Agent header
HEADERS = {
    "User-Agent": "Najad Marathumpalli mrnajadas@gmail.com"
}

COMPANIES = {
    "AAPL": {"name": "Apple",             "cik": "320193"},
    "MSFT": {"name": "Microsoft",         "cik": "789019"},
    "JPM":  {"name": "JPMorgan Chase",    "cik": "19617"},
    "XOM":  {"name": "ExxonMobil",        "cik": "34088"},
    "JNJ":  {"name": "Johnson & Johnson", "cik": "200406"},
}


def get_10k_url(cik: str) -> str:
    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i]
            primary_doc = filings["primaryDocument"][i]
            accession_clean = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{accession_clean}/{primary_doc}"
            )

    raise ValueError(f"No 10-K filing found for CIK {cik}")


def download_and_parse(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def save_text(text: str, company_name: str) -> str:
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

    print("Downloading and parsing HTML (this may take a minute)...")
    text = download_and_parse(url)

    filepath = save_text(text, company["name"])
    print(f"Done. Saved {len(text):,} characters to {filepath}")


if __name__ == "__main__":
    main()
