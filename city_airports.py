"""
city_airports.py

Maps city/metro codes to their constituent airport IATA codes.
Used to run multi-airport searches and merge results for major cities.

When a user searches LON, we search LHR + LGW + LTN and merge.
When a user searches NYC, we search JFK + EWR + LGA and merge.
"""

CITY_AIRPORTS = {
    # UK
    "LON": ["LHR", "LGW", "LTN", "STN", "LCY"],
    "MAN": ["MAN"],
    "BHX": ["BHX"],
    # Europe
    "PAR": ["CDG", "ORY"],
    "AMS": ["AMS"],
    "FRA": ["FRA"],
    "MAD": ["MAD"],
    "BCN": ["BCN"],
    "FCO": ["FCO", "CIA"],
    "MXP": ["MXP", "LIN"],
    "ZRH": ["ZRH"],
    "VIE": ["VIE"],
    "BRU": ["BRU"],
    "CPH": ["CPH"],
    "OSL": ["OSL"],
    "ARN": ["ARN"],
    "HEL": ["HEL"],
    "LIS": ["LIS"],
    "ATH": ["ATH"],
    "IST": ["IST", "SAW"],
    "DXB": ["DXB"],
    "AUH": ["AUH"],
    # North America
    "NYC": ["JFK", "EWR", "LGA"],
    "LON": ["LHR", "LGW", "LTN", "STN", "LCY"],
    "LAX": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "CHI": ["ORD", "MDW"],
    "MIA": ["MIA", "FLL"],
    "SFO": ["SFO", "OAK", "SJC"],
    "WAS": ["IAD", "DCA", "BWI"],
    "BOS": ["BOS"],
    "YTO": ["YYZ", "YTZ"],
    "YMQ": ["YUL"],
    # Asia Pacific
    "TYO": ["NRT", "HND"],
    "OSA": ["KIX", "ITM"],
    "SEL": ["ICN", "GMP"],
    "BJS": ["PEK", "PKX"],
    "SHA": ["PVG", "SHA"],
    "HKG": ["HKG"],
    "SIN": ["SIN"],
    "BKK": ["BVK", "DMK"],
    "KUL": ["KUL"],
    "SYD": ["SYD"],
    "MEL": ["MEL"],
    # Middle East / Africa
    "TLV": ["TLV"],
    "CAI": ["CAI"],
    "JNB": ["JNB"],
    "CPT": ["CPT"],
    "NBO": ["NBO"],
    # South America
    "GRU": ["GRU", "CGH"],
    "EZE": ["EZE", "AEP"],
    "SCL": ["SCL"],
    "BOG": ["BOG"],
    "LIM": ["LIM"],
}


def get_airports_for_code(code: str) -> list:
    """
    Given an airport or city code, return list of airports to search.
    If it's a known city code, returns all airports.
    If it's a direct airport code, returns just that code.
    Always deduplicates and returns at least the original code.
    """
    if not code:
        return []
    code = code.upper().strip()
    airports = CITY_AIRPORTS.get(code)
    if airports:
        return list(dict.fromkeys(airports))  # deduplicated, order preserved
    # Not a known city code — treat as direct airport code
    return [code]


def is_city_code(code: str) -> bool:
    return code.upper().strip() in CITY_AIRPORTS
