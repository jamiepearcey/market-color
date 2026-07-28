# /// script
# requires-python = ">=3.10"
# ///
"""Curated ticker -> GICS sector map for the eg100k 2010-2012 universe.

Yahoo's assetProfile (the automatic source) now 401s without a crumb, so this is
a hand-classified map of the recognizable, US-tradable names — the ones that
actually survive price-residualisation and land in baskets. Obscure OTC/ADR
tickers are deliberately left out (SECTOR() returns "UNK"); they drop out of the
returns panel anyway. Payment networks (MA/V) follow the 2023 GICS move to
Financials. This is used for the honest GICS basket control + sector-spread
report, NOT for neutralisation (that stays on the 9 SPDR sector ETFs)."""

_GICS = {
    # Financials — banks, insurers, asset managers, GSEs, exchanges, payments
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "C": "Financials",
    "MS": "Financials", "WFC": "Financials", "AIG": "Financials", "FNMA": "Financials",
    "FMCC": "Financials", "AGNC": "Financials", "BLK": "Financials", "BNY": "Financials",
    "MET": "Financials", "CG": "Financials", "ALLY": "Financials", "KEY": "Financials",
    "CME": "Financials", "UBS": "Financials", "DB": "Financials", "HSBC": "Financials",
    "BCS": "Financials", "LYG": "Financials", "NWG": "Financials", "MUFG": "Financials",
    "OTPBF": "Financials", "EBKOF": "Financials", "ACGBY": "Financials", "SCBFF": "Financials",
    "WF": "Financials", "TD": "Financials", "DBOEY": "Financials", "SGBLY": "Financials",
    "FRFHF": "Financials", "LEHNQ": "Financials", "BRK-B": "Financials", "MA": "Financials",
    "TMXXF": "Financials",
    # Information Technology
    "AAPL": "InfoTech", "MSFT": "InfoTech", "ORCL": "InfoTech", "IBM": "InfoTech",
    "INTC": "InfoTech", "SAP": "InfoTech", "HPE": "InfoTech", "MU": "InfoTech",
    "ADBE": "InfoTech", "CIEN": "InfoTech", "XRX": "InfoTech", "BB": "InfoTech",
    "LPL": "InfoTech", "SPWR": "InfoTech", "CAJPY": "InfoTech",
    # Communication Services
    "GOOGL": "CommSvcs", "META": "CommSvcs", "NFLX": "CommSvcs", "T": "CommSvcs",
    "VOD": "CommSvcs", "DIS": "CommSvcs", "NWSA": "CommSvcs", "NYT": "CommSvcs",
    "DTEGY": "CommSvcs", "TELFY": "CommSvcs", "PTNRF": "CommSvcs", "MTNOY": "CommSvcs",
    "SINGY": "CommSvcs", "BCE": "CommSvcs", "MANU": "CommSvcs", "BIDU": "CommSvcs",
    "RMVEY": "CommSvcs", "SWGNF": "CommSvcs",
    # Energy
    "XOM": "Energy", "BP": "Energy", "SHEL": "Energy", "PBR": "Energy", "PTBRY": "Energy",
    "ENB": "Energy", "BKR": "Energy", "LNG": "Energy", "OMVKY": "Energy", "TK": "Energy",
    # Industrials
    "GE": "Industrials", "BA": "Industrials", "UAL": "Industrials", "AAL": "Industrials",
    "DAL": "Industrials", "LMT": "Industrials", "SIEGY": "Industrials", "CP": "Industrials",
    "BAESY": "Industrials", "VWSYF": "Industrials", "RTNTF": "Industrials",
    # Consumer Discretionary
    "GM": "ConsDisc", "F": "ConsDisc", "TM": "ConsDisc", "HMC": "ConsDisc",
    "NSANY": "ConsDisc", "VWAGY": "ConsDisc", "MBGAF": "ConsDisc", "LVS": "ConsDisc",
    "AMZN": "ConsDisc", "UAA": "ConsDisc", "SONY": "ConsDisc",
    # Consumer Staples
    "WMT": "Staples", "PM": "Staples", "TSN": "Staples", "CLX": "Staples",
    # Health Care
    "PFE": "Health", "TEVA": "Health", "GILD": "Health", "MDT": "Health", "JNJ": "Health",
    # Materials
    "BHP": "Materials", "AA": "Materials", "VALE": "Materials", "SCCO": "Materials",
    "AU": "Materials", "GFI": "Materials", "HMY": "Materials", "KUMBF": "Materials",
    "AAUKF": "Materials", "ZIJMY": "Materials", "PAANF": "Materials",
    # Utilities
    "TKECY": "Utilities", "PCG": "Utilities", "VEOEY": "Utilities",
}


def sector(t):
    return _GICS.get(t, "UNK")


def known(names):
    return {s: _GICS[s] for s in names if s in _GICS}
