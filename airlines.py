# airlines.py

# =====================================================================
# SECTION START: AIRLINE NAMES
# IATA code -> human-readable airline name
# Used by map_ttn_offer_to_option and map_duffel_offer_to_option
# as a fallback when the provider does not supply a name directly.
# =====================================================================

AIRLINE_NAMES = {
    # Europe — full-service
    "LH": "Lufthansa",
    "BA": "British Airways",
    "AF": "Air France",
    "KL": "KLM",
    "LX": "Swiss",
    "SN": "Brussels Airlines",
    "OS": "Austrian Airlines",
    "SK": "Scandinavian Airlines",
    "AZ": "ITA Airways",
    "LO": "LOT Polish Airlines",
    "TP": "TAP Air Portugal",
    "IB": "Iberia",
    "TK": "Turkish Airlines",
    "LG": "Luxair",
    "OA": "Olympic Air",
    "A3": "Aegean Airlines",
    "JU": "Air Serbia",
    "RO": "TAROM",
    "OK": "Czech Airlines",
    "BT": "airBaltic",
    "FB": "Bulgaria Air",
    "HV": "Transavia",
    "TO": "Transavia France",
    "HG": "Niki",
    "EI": "Aer Lingus",
    "AY": "Finnair",
    "WF": "Widerøe",
    "DY": "Norwegian",
    "D8": "Norwegian International",
    "IBE": "Iberia Express",
    "I2": "Iberia Express",
    "VY": "Vueling",
    "VS": "Virgin Atlantic",
    "BE": "Flybe",
    "LS": "Jet2",
    "X3": "TUIfly",
    "DS": "easyJet Switzerland",
    "U2": "easyJet",
    "FR": "Ryanair",
    "W6": "Wizz Air",
    "4U": "Germanwings",
    "EW": "Eurowings",
    "DE": "Condor",
    "X9": "Avion Express",
    "MT": "Thomas Cook Airlines",
    "BY": "TUI Airways",
    "OR": "TUI fly Netherlands",
    "TB": "TUI fly Belgium",
    "MF": "Xiamen Airlines",  # codeshare on European routes

    # Middle East
    "LY": "El Al",
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "EY": "Etihad Airways",
    "SV": "Saudia",
    "RJ": "Royal Jordanian",
    "WY": "Oman Air",
    "GF": "Gulf Air",
    "ME": "Middle East Airlines",
    "IR": "Iran Air",
    "FZ": "flydubai",
    "G9": "Air Arabia",
    "XY": "flynas",
    "PC": "Pegasus Airlines",
    "XQ": "SunExpress",

    # North America
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "B6": "JetBlue Airways",
    "WS": "WestJet",
    "AC": "Air Canada",
    "F9": "Frontier Airlines",
    "NK": "Spirit Airlines",
    "AS": "Alaska Airlines",
    "HA": "Hawaiian Airlines",
    "WN": "Southwest Airlines",
    "SY": "Sun Country Airlines",
    "G4": "Allegiant Air",
    "PD": "Porter Airlines",

    # Latin America
    "LA": "LATAM Airlines",
    "AV": "Avianca",
    "AM": "Aeromexico",
    "CM": "Copa Airlines",
    "G3": "Gol Linhas Aéreas",
    "UX": "Air Europa",
    "H2": "Sky Airline",
    "JA": "JetSMART",
    "P9": "Peruvian Airlines",

    # Africa
    "ET": "Ethiopian Airlines",
    "KQ": "Kenya Airways",
    "MS": "EgyptAir",
    "AT": "Royal Air Maroc",
    "TU": "Tunisair",
    "SA": "South African Airways",
    "HM": "Air Seychelles",
    "WB": "RwandAir",
    "TC": "Air Tanzania",
    "UG": "Tunisair Express",
    "HC": "Air Senegal",
    "VT": "Air Tahiti",

    # Asia — full-service
    "SQ": "Singapore Airlines",
    "CX": "Cathay Pacific",
    "JL": "Japan Airlines",
    "NH": "ANA All Nippon Airways",
    "KE": "Korean Air",
    "OZ": "Asiana Airlines",
    "CI": "China Airlines",
    "BR": "EVA Air",
    "MU": "China Eastern",
    "CA": "Air China",
    "CZ": "China Southern",
    "HX": "Hong Kong Airlines",
    "TG": "Thai Airways",
    "VN": "Vietnam Airlines",
    "MH": "Malaysia Airlines",
    "GA": "Garuda Indonesia",
    "PR": "Philippine Airlines",
    "AI": "Air India",
    "UK": "Vistara",
    "IX": "Air India Express",
    "6E": "IndiGo",
    "SG": "SpiceJet",
    "I5": "AirAsia India",
    "TR": "Scoot",
    "D7": "AirAsia X",
    "AK": "AirAsia",
    "FD": "Thai AirAsia",
    "Z2": "Philippines AirAsia",
    "QZ": "Indonesia AirAsia",
    "XT": "Indonesia AirAsia X",
    "OD": "Malindo Air",
    "MI": "SilkAir",
    "3K": "Jetstar Asia",
    "JQ": "Jetstar",
    "GK": "Jetstar Japan",
    "BX": "Air Busan",
    "TW": "T'way Air",
    "LJ": "Jin Air",
    "RS": "Air Seoul",
    "ZE": "Eastar Jet",
    "7C": "Jeju Air",

    # Oceania
    "QF": "Qantas",
    "NZ": "Air New Zealand",
    "VA": "Virgin Australia",
    "FJ": "Fiji Airways",
    "PX": "Air Niugini",

    # Central Asia / Former Soviet
    "KC": "Air Astana",
    "PS": "Ukraine International",
    "UR": "UTair",
    "SU": "Aeroflot",
    "S7": "S7 Airlines",
    "DP": "Pobeda",
    "5N": "Nordavia",
    "U6": "Ural Airlines",
    "N4": "Nordwind Airlines",

    # Low-cost / charter extras
    "XR": "Corendon Airlines",
    "XC": "Corendon Dutch Airlines",
    "7G": "Star Flyer",
    "MM": "Peach Aviation",
    "HB": "Greater Bay Airlines",
}

# =====================================================================
# SECTION END: AIRLINE NAMES
# =====================================================================


# =====================================================================
# SECTION START: AIRLINE BOOKING URLS
# IATA code -> airline direct booking URL
# Used to generate the "Book direct" link on flight cards.
# =====================================================================

AIRLINE_BOOKING_URLS = {
    "LH": "https://www.lufthansa.com/gb/en/flight-search",
    "BA": "https://www.britishairways.com/travel/home/public/en_gb",
    "AF": "https://wwws.airfrance.co.uk/",
    "KL": "https://www.klm.co.uk/",
    "LX": "https://www.swiss.com/gb/en/homepage",
    "SN": "https://www.brusselsairlines.com/",
    "OS": "https://www.austrian.com/",
    "SK": "https://www.flysas.com/en/",
    "AZ": "https://www.ita-airways.com/en_gb",
    "LO": "https://www.lot.com/gb/en",
    "TP": "https://www.flytap.com/en-gb/",
    "IB": "https://www.iberia.com/gb/",
    "VY": "https://www.vueling.com/en",
    "U2": "https://www.easyjet.com/en",
    "FR": "https://www.ryanair.com/gb/en",
    "W6": "https://wizzair.com/",
    "EI": "https://www.aerlingus.com/",
    "AY": "https://www.finnair.com/",
    "BT": "https://www.airbaltic.com/en/",
    "OK": "https://www.csa.cz/en/",
    "RO": "https://www.tarom.ro/en",
    "JU": "https://www.airserbia.com/en",
    "FB": "https://www.air.bg/en",
    "HV": "https://www.transavia.com/en-UK/home/",
    "DS": "https://www.easyjet.com/en",
    "A3": "https://en.aegeanair.com/",
    "TK": "https://www.turkishairlines.com/en-int/flights/",
    "LG": "https://www.luxair.lu/en",
    "X3": "https://www.tuifly.com/",
    "LS": "https://www.jet2.com/",
    "VS": "https://www.virginatlantic.com/",
    "EW": "https://www.eurowings.com/en.html",
    "DE": "https://www.condor.com/gb/",
    "BY": "https://www.tui.co.uk/destinations/flights",
    "DY": "https://www.norwegian.com/uk/",
    "PC": "https://www.flypgs.com/en",
    "XQ": "https://www.sunexpress.com/en/",
    "FZ": "https://www.flydubai.com/en/",
    "G9": "https://www.airarabia.com/en",
    "LY": "https://www.elal.com/en/",
    "EK": "https://www.emirates.com/uk/english/",
    "QR": "https://www.qatarairways.com/en-gb/homepage.html",
    "EY": "https://www.etihad.com/en-gb/",
    "SV": "https://www.saudia.com/",
    "RJ": "https://www.rj.com/",
    "WY": "https://www.omanair.com/",
    "GF": "https://www.gulfair.com/",
    "ME": "https://www.mea.com.lb/english",
    "AA": "https://www.aa.com/",
    "DL": "https://www.delta.com/",
    "UA": "https://www.united.com/",
    "B6": "https://www.jetblue.com/",
    "WS": "https://www.westjet.com/",
    "AC": "https://www.aircanada.com/",
    "F9": "https://www.flyfrontier.com/",
    "NK": "https://www.spirit.com/",
    "AS": "https://www.alaskaair.com/",
    "HA": "https://www.hawaiianairlines.com/",
    "LA": "https://www.latamairlines.com/",
    "AV": "https://www.avianca.com/",
    "AM": "https://aeromexico.com/en-gb",
    "CM": "https://www.copaair.com/",
    "G3": "https://www.voegol.com.br/en",
    "UX": "https://www.aireuropa.com/",
    "SQ": "https://www.singaporeair.com/",
    "CX": "https://www.cathaypacific.com/",
    "JL": "https://www.jal.co.jp/jp/en/",
    "NH": "https://www.ana.co.jp/en/jp/",
    "KE": "https://www.koreanair.com/",
    "OZ": "https://flyasiana.com/",
    "CI": "https://www.china-airlines.com/",
    "BR": "https://www.evaair.com/",
    "MU": "https://www.ceair.com/",
    "CA": "https://www.airchina.com/",
    "CZ": "https://www.csair.com/",
    "HX": "https://www.hongkongairlines.com/",
    "TG": "https://www.thaiairways.com/",
    "VN": "https://www.vietnamairlines.com/",
    "MH": "https://www.malaysiaairlines.com/",
    "GA": "https://www.garuda-indonesia.com/",
    "PR": "https://www.philippineairlines.com/",
    "AI": "https://www.airindia.com/",
    "UK": "https://www.airvistara.com/",
    "6E": "https://www.goindigo.in/",
    "SG": "https://www.spicejet.com/",
    "TR": "https://www.flyscoot.com/",
    "D7": "https://www.airasia.com/",
    "AK": "https://www.airasia.com/",
    "QF": "https://www.qantas.com/",
    "NZ": "https://www.airnewzealand.co.uk/",
    "VA": "https://www.virginaustralia.com/",
    "JQ": "https://www.jetstar.com/",
    "ET": "https://www.ethiopianairlines.com/",
    "KQ": "https://www.kenya-airways.com/",
    "MS": "https://www.egyptair.com/",
    "AT": "https://www.royalairmaroc.com/",
    "TU": "https://www.tunisair.com/",
    "SA": "https://www.flysaa.com/",
    "HM": "https://www.airseychelles.com/",
    "WB": "https://www.rwandair.com/",
    "XR": "https://www.corendonairlines.com/",
    "KC": "https://www.airastana.com/",
    "SU": "https://www.aeroflot.ru/en",
    "S7": "https://www.s7.ru/en/",
}

# =====================================================================
# SECTION END: AIRLINE BOOKING URLS
# =====================================================================


# =====================================================================
# SECTION START: AIRCRAFT NAMES
# ICAO equipment code -> human-readable aircraft name
# TTN returns ICAO codes in segment data (e.g. "7M8", "789", "73H").
# Used in map_ttn_offer_to_option to populate aircraftName on segments.
# =====================================================================

AIRCRAFT_NAMES = {
    # Boeing 737 family
    "733": "Boeing 737-300",
    "734": "Boeing 737-400",
    "735": "Boeing 737-500",
    "736": "Boeing 737-600",
    "737": "Boeing 737-700",
    "738": "Boeing 737-800",
    "739": "Boeing 737-900",
    "73H": "Boeing 737-800",
    "73J": "Boeing 737-900",
    "73W": "Boeing 737-700",
    "73C": "Boeing 737-300 Combi",
    "7M7": "Boeing 737 MAX 7",
    "7M8": "Boeing 737 MAX 8",
    "7M9": "Boeing 737 MAX 9",
    "7MJ": "Boeing 737 MAX 10",

    # Boeing 747 family
    "741": "Boeing 747-100",
    "742": "Boeing 747-200",
    "743": "Boeing 747-300",
    "744": "Boeing 747-400",
    "74E": "Boeing 747-400 Combi",
    "74F": "Boeing 747-400F",
    "748": "Boeing 747-8",
    "74H": "Boeing 747-8",

    # Boeing 757 family
    "752": "Boeing 757-200",
    "753": "Boeing 757-300",
    "75W": "Boeing 757-200",

    # Boeing 767 family
    "762": "Boeing 767-200",
    "763": "Boeing 767-300",
    "764": "Boeing 767-400",
    "76W": "Boeing 767-300",

    # Boeing 777 family
    "772": "Boeing 777-200",
    "77L": "Boeing 777-200LR",
    "773": "Boeing 777-300",
    "77W": "Boeing 777-300ER",
    "778": "Boeing 777X-8",
    "779": "Boeing 777X-9",

    # Boeing 787 family
    "787": "Boeing 787 Dreamliner",
    "788": "Boeing 787-8 Dreamliner",
    "789": "Boeing 787-9 Dreamliner",
    "78J": "Boeing 787-10 Dreamliner",

    # Airbus A220 family (formerly Bombardier C Series)
    "221": "Airbus A220-100",
    "223": "Airbus A220-300",
    "CS1": "Airbus A220-100",
    "CS3": "Airbus A220-300",

    # Airbus A319/320/321 family
    "319": "Airbus A319",
    "320": "Airbus A320",
    "321": "Airbus A321",
    "31X": "Airbus A318/A319",
    "32A": "Airbus A320neo",
    "32B": "Airbus A321neo",
    "32N": "Airbus A320neo",
    "32Q": "Airbus A321neo",
    "32S": "Airbus A321",

    # Airbus A330 family
    "330": "Airbus A330",
    "332": "Airbus A330-200",
    "333": "Airbus A330-300",
    "338": "Airbus A330-800neo",
    "339": "Airbus A330-900neo",
    "33X": "Airbus A330-200",

    # Airbus A340 family
    "340": "Airbus A340",
    "342": "Airbus A340-200",
    "343": "Airbus A340-300",
    "345": "Airbus A340-500",
    "346": "Airbus A340-600",

    # Airbus A350 family
    "350": "Airbus A350 XWB",
    "351": "Airbus A350-900",
    "358": "Airbus A350-800",
    "359": "Airbus A350-900",
    "35K": "Airbus A350-1000",

    # Airbus A380
    "380": "Airbus A380",
    "388": "Airbus A380-800",

    # ATR family (regional turboprop)
    "AT4": "ATR 42",
    "AT5": "ATR 42-500",
    "AT7": "ATR 72",
    "ATR": "ATR 72",

    # Bombardier / Embraer regional jets
    "CRJ": "Bombardier CRJ",
    "CR1": "Bombardier CRJ-100",
    "CR2": "Bombardier CRJ-200",
    "CR7": "Bombardier CRJ-700",
    "CR9": "Bombardier CRJ-900",
    "CRA": "Bombardier CRJ-1000",
    "E70": "Embraer E170",
    "E75": "Embraer E175",
    "E90": "Embraer E190",
    "E95": "Embraer E195",
    "E7W": "Embraer E175-E2",
    "E7J": "Embraer E190-E2",
    "E7K": "Embraer E195-E2",
    "ERJ": "Embraer ERJ",
    "ER3": "Embraer ERJ-135",
    "ER4": "Embraer ERJ-145",

    # Sukhoi / Irkut
    "SU9": "Sukhoi Superjet 100",
    "MC2": "Irkut MC-21",

    # Misc
    "DH4": "De Havilland Q400",
    "DH8": "De Havilland Dash 8",
    "F70": "Fokker 70",
    "F100": "Fokker 100",
    "SF3": "Saab SF340",
}

# =====================================================================
# SECTION END: AIRCRAFT NAMES
# =====================================================================
