

# Universe of stocks sorted by GICS Sectors
UNIVERSE = {
    "Information Technology": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # NVIDIA
        "AVGO",   # Broadcom
        "CRM",    # Salesforce
        "ADBE",   # Adobe
        "INTC",   # Intel
    ],
    "Health Care": [
        "UNH",    # UnitedHealth
        "JNJ",    # Johnson & Johnson
        "PFE",    # Pfizer
        "ABT",    # Abbott Labs
        "TMO",    # Thermo Fisher
        "MRK",    # Merck
        "LLY",    # Eli Lilly
    ],
    "Financials": [
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "BLK",    # BlackRock
        "C",      # Citigroup
        "AXP",    # American Express
    ],
    "Consumer Discretionary": [
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        "DIS",    # Disney
        "HD",     # Home Depot
        "NKE",    # Nike
        "MCD",    # McDonald's
        "LOW",    # Lowe's
    ],
    "Consumer Staples": [
        "WMT",    # Walmart
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "COST",   # Costco
        "CL",     # Colgate-Palmolive
    ],
    "Energy": [
        "XOM",    # ExxonMobil
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger
        "EOG",    # EOG Resources
        "PSX",    # Phillips 66
    ],
    "Industrials": [
        "CAT",    # Caterpillar
        "HON",    # Honeywell
        "UPS",    # United Parcel Service
        "BA",     # Boeing
        "RTX",    # RTX (Raytheon)
        "DE",     # Deere & Co
        "GE",     # GE Aerospace
    ],
    "Communication Services": [
        "GOOG",   # Alphabet
        "META",   # Meta Platforms
        "NFLX",   # Netflix
        "CMCSA",  # Comcast
        "T",      # AT&T
        "VZ",     # Verizon
    ],
    "Utilities": [
        "NEE",    # NextEra Energy
        "DUK",    # Duke Energy
        "SO",     # Southern Company
        "D",      # Dominion Energy
        "AEP",    # American Electric Power
        "SRE",    # Sempra
    ],
    "Real Estate": [
        "PLD",    # Prologis
        "AMT",    # American Tower
        "CCI",    # Crown Castle
        "SPG",    # Simon Property Group
        "PSA",    # Public Storage
        "EQIX",   # Equinix
    ],
    "Materials": [
        "LIN",    # Linde
        "APD",    # Air Products
        "SHW",    # Sherwin-Williams
        "FCX",    # Freeport-McMoRan
        "NEM",    # Newmont
        "ECL",    # Ecolab
    ],
}

# Flat list of all tickers in the universe
ALL_TICKERS = [ticker for sector in UNIVERSE.values() for ticker in sector]

# Stocks we actually generate forecasts for
# Includes all 7 original strategy tickers + additional liquid names
FORECAST_TICKERS = [

    "AAPL", "DIS", "MSFT", "PFE", "UNH", "WMT", "XOM", "AMZN", "GOOG", "JPM", "GS", "NVDA", "TSLA", "BA",
    "JNJ", "HD", "META",
]

# Lookups - for sanity sake and later checking if required
_TICKER_TO_SECTOR = {
    ticker: sector
    for sector, tickers in UNIVERSE.items()
    for ticker in tickers
}


def get_sector(ticker: str) -> str:
    """Return the GICS sector for a given ticker."""
    return _TICKER_TO_SECTOR.get(ticker.upper(), "Unknown")


def get_universe_summary() -> dict:
    """Print a quick summary of the universe."""
    return {
        "total_stocks": len(ALL_TICKERS),
        "sectors": len(UNIVERSE),
        "forecast_subset": len(FORECAST_TICKERS),
        "stocks_per_sector": {sector: len(tickers) for sector, tickers in UNIVERSE.items()},
    }


if __name__ == "__main__":
    summary = get_universe_summary()
    print(
        f"Universe: {summary['total_stocks']} stocks across {summary['sectors']} sectors")
    print(f"Forecast subset: {summary['forecast_subset']} stocks")
    print()
    for sector, count in summary["stocks_per_sector"].items():
        tickers = UNIVERSE[sector]
        print(f"  {sector} ({count}): {', '.join(tickers)}")
