import re

# ──────────────────────────────────────────
# URDU TO ROMAN URDU TRANSLITERATION v4
# Social media style — based on:
# - Roman Urdu Wikipedia conventions
# - Pakistani Twitter/social media patterns
# - BGN/PCGN 2007 standard for named entities
# - Verified common word mappings from
#   Urdu language learning resources
# ──────────────────────────────────────────


# ── PRIORITY 1: Function Word Dictionary ─
# Most frequent Urdu words with their exact
# social media Roman Urdu spellings.
# These appear in nearly every sentence and
# must be correct for readable output.

FUNCTION_WORDS = {
    # ── Pronouns ──
    "میں":      "mein",
    "مجھے":     "mujhe",
    "مجھ":      "mujh",
    "ہم":       "hum",
    "ہمارا":    "hamara",
    "ہماری":    "hamari",
    "ہمارے":    "hamare",
    "آپ":       "aap",
    "تم":       "tum",
    "تمہارا":   "tumhara",
    "تمہاری":   "tumhari",
    "تو":       "tu",
    "وہ":       "woh",
    "اس":       "is",
    "اسے":      "ise",
    "اسی":      "isi",
    "انہیں":    "unhen",
    "انہوں":    "unhon",
    "ان":       "in",
    "انکا":     "unka",
    "انکی":     "unki",
    "انکے":     "unke",
    "یہ":       "yeh",
    "اسکا":     "uska",
    "اسکی":     "uski",
    "اسکے":     "uske",
    "جو":       "jo",
    "جس":       "jis",
    "جن":       "jin",
    "جسے":      "jise",
    "جنہیں":    "jinhen",
    "کون":      "kaun",
    "کیا":      "kya",
    "کس":       "kis",
    "کوئی":     "koi",
    "کچھ":      "kuch",
    "سب":       "sab",
    "ہر":       "har",
    "کئی":      "kai",
    "خود":      "khud",
    "اپنا":     "apna",
    "اپنی":     "apni",
    "اپنے":     "apne",

    # ── Postpositions / case markers ──
    "نے":       "ne",
    "کا":       "ka",
    "کی":       "ki",
    "کے":       "ke",
    "کو":       "ko",
    "سے":       "se",
    "میں":      "mein",
    "پر":       "par",
    "تک":       "tak",
    "ساتھ":     "saath",
    "لیے":      "liye",
    "لئے":      "liye",
    "بارے":     "baray",
    "بارہ":     "baara",
    "خلاف":     "khilaf",
    "لیکر":     "lekar",
    "اندر":     "andar",
    "باہر":     "bahar",
    "اوپر":     "upar",
    "نیچے":     "neeche",
    "آگے":      "aage",
    "پیچھے":    "peeche",
    "قریب":     "qareeb",
    "دوران":    "dauran",
    "بعد":      "baad",
    "پہلے":     "pehle",
    "جانب":     "janib",

    # ── Conjunctions ──
    "اور":      "aur",
    "و":        "aur",
    "یا":       "ya",
    "لیکن":     "lekin",
    "مگر":      "magar",
    "کہ":       "ke",
    "کہا":      "kaha",
    "کہنا":     "kehna",
    "جب":       "jab",
    "تو":       "to",
    "اگر":      "agar",
    "کیونکہ":   "kyunke",
    "تاکہ":     "taake",
    "جبکہ":     "jabke",
    "حالانکہ":  "halanke",
    "البتہ":    "albatta",
    "ورنہ":     "warna",
    "پھر":      "phir",
    "ابھی":     "abhi",
    "اب":       "ab",
    "پھر":      "phir",
    "بھی":      "bhi",
    "ہی":       "hi",
    "تو":       "to",
    "ہاں":      "haan",
    "نہیں":     "nahi",
    "نہ":       "na",
    "نا":       "na",
    "جی":       "ji",

    # ── Auxiliary verbs / tense markers ──
    "ہے":       "hai",
    "ہیں":      "hain",
    "ہو":       "ho",
    "ہوں":      "hoon",
    "ہونا":     "hona",
    "ہوتا":     "hota",
    "ہوتی":     "hoti",
    "ہوتے":     "hote",
    "ہوا":      "hua",
    "ہوئی":     "hui",
    "ہوئے":     "hue",
    "ہوگا":     "hoga",
    "ہوگی":     "hogi",
    "ہوں گے":   "hon ge",
    "تھا":      "tha",
    "تھی":      "thi",
    "تھے":      "the",
    "گا":       "ga",
    "گی":       "gi",
    "گے":       "ge",
    "کرنا":     "karna",
    "کرتا":     "karta",
    "کرتی":     "karti",
    "کرتے":     "karte",
    "کرے":      "kare",
    "کریں":     "karen",
    "کیا":      "kiya",
    "کی":       "ki",
    "کر":       "kar",
    "کرکے":     "karke",
    "کرنے":     "karne",
    "جانا":     "jana",
    "جائے":     "jaye",
    "جائیں":    "jayen",
    "جاتا":     "jaata",
    "جاتی":     "jaati",
    "جاتے":     "jaate",
    "گیا":      "gaya",
    "گئی":      "gayi",
    "گئے":      "gaye",
    "آنا":      "aana",
    "آئے":      "aaye",
    "آئی":      "aayi",
    "آیا":      "aaya",
    "آ":        "aa",
    "جا":       "ja",
    "دینا":     "dena",
    "دیا":      "diya",
    "دی":       "di",
    "دے":       "de",
    "دیں":      "den",
    "دیکھنا":   "dekhna",
    "دیکھا":    "dekha",
    "دیکھے":    "dekhe",
    "لینا":     "lena",
    "لیا":      "liya",
    "لی":       "li",
    "لے":       "le",
    "رہنا":     "rehna",
    "رہا":      "raha",
    "رہی":      "rahi",
    "رہے":      "rahe",
    "رہیں":     "rahen",
    "ملنا":     "milna",
    "ملا":      "mila",
    "ملی":      "mili",
    "ملے":      "mile",
    "بتانا":    "batana",
    "بتایا":    "bataya",
    "چاہنا":    "chahna",
    "چاہیے":    "chahiye",
    "چاہتا":    "chahta",
    "چاہتی":    "chahti",
    "چاہتے":    "chahte",
    "سکنا":     "sakna",
    "سکتا":     "sakta",
    "سکتی":     "sakti",
    "سکتے":     "sakte",
    "سکا":      "saka",
    "سکی":      "saki",
    "سکے":      "sake",
    "مل":       "mil",
    "لگنا":     "lagna",
    "لگا":      "laga",
    "لگی":      "lagi",
    "لگے":      "lage",
    "لگتا":     "lagta",

    # ── Common adjectives / adverbs ──
    "بہت":      "bohot",
    "زیادہ":    "zyada",
    "کم":       "kam",
    "اچھا":     "acha",
    "اچھی":     "achi",
    "اچھے":     "ache",
    "برا":      "bura",
    "بری":      "buri",
    "بڑا":      "bara",
    "بڑی":      "bari",
    "بڑے":      "bare",
    "چھوٹا":    "chota",
    "چھوٹی":    "choti",
    "نیا":      "naya",
    "نئی":      "nayi",
    "نئے":      "naye",
    "پرانا":    "purana",
    "صرف":      "sirf",
    "پھر":      "phir",
    "ابھی":     "abhi",
    "ہاں":      "haan",
    "واقعی":    "waqai",
    "اکثر":     "aksar",
    "کبھی":     "kabhi",
    "کبھی کبھی": "kabhi kabhi",
    "ہمیشہ":    "hamesha",
    "کبھی نہیں": "kabhi nahi",
    "پھر بھی":  "phir bhi",
    "اس لیے":   "is liye",
    "اس لئے":   "is liye",

    # ── Common nouns (high frequency) ──
    "بات":      "baat",
    "باتیں":    "baatein",
    "وقت":      "waqt",
    "کام":      "kaam",
    "جگہ":      "jagah",
    "دن":       "din",
    "رات":      "raat",
    "سال":      "saal",
    "ملک":      "mulk",
    "لوگ":      "log",
    "آدمی":     "aadmi",
    "عورت":     "aurat",
    "بچہ":      "bacha",
    "بچے":      "bachay",
    "گھر":      "ghar",
    "خاندان":   "khandan",
    "حکومت":    "hukumat",
    "عدالت":    "adalat",
    "معاملہ":   "mamla",
    "معاملے":   "mamlay",
    "طرف":      "taraf",
    "حصہ":      "hissa",
    "جگہ":      "jagah",
    "جگہوں":    "jagahon",
    "خبر":      "khabar",
    "خبریں":    "khabrein",
    "نام":      "naam",
    "بیان":     "bayan",
    "فیصلہ":    "faisla",
    "الزام":    "ilzam",
    "اعلان":    "elaan",
    "تقریر":    "taqreer",
    "انتخاب":   "intikhab",
    "اختیار":   "ikhtiyar",
    "استعمال":  "istemal",
    "ناجائز":   "najayaz",
    "سابق":     "sabiq",
    "نائب":     "naib",
}


# ── PRIORITY 2: English Loanwords ────────
ENGLISH_LOANWORDS = {
    "ڈاکٹر":    "doctor",
    "پولیس":    "police",
    "سپیکر":    "speaker",
    "پارٹی":    "party",
    "جسٹس":     "justice",
    "کورٹ":     "court",
    "سپریم":    "supreme",
    "پارلیمنٹ": "parliament",
    "سینیٹ":    "senate",
    "اسمبلی":   "assembly",
    "الیکشن":   "election",
    "کمیٹی":    "committee",
    "میڈیا":    "media",
    "ریڈیو":    "radio",
    "انٹرویو":  "interview",
    "رپورٹ":    "report",
    "ایجنسی":   "agency",
    "پریس":     "press",
    "ٹیم":      "team",
    "کوچ":      "coach",
    "بجٹ":      "budget",
    "بینک":     "bank",
    "مارکیٹ":   "market",
    "پروگرام":  "program",
    "کمانڈر":   "commander",
    "انٹرنیشنل": "international",
    "ٹی وی":    "tv",
    "ٹی":       "t",
    "وی":       "v",
    "ڈی":       "d",
    "کیو":      "q",
    "اے":       "a",
    "ایم":      "m",
    "پی":       "p",
    "ایس":      "s",
}


# ── PRIORITY 3: Named Entity Dictionary ──
NAMED_ENTITY_DICT = {
    # Locations
    "پاکستان":    "pakistan",
    "لاہور":      "lahore",
    "کراچی":      "karachi",
    "اسلام":      "islam",
    "آباد":       "abad",
    "اسلام آباد": "islamabad",
    "پشاور":      "peshawar",
    "کوئٹہ":      "quetta",
    "ملتان":      "multan",
    "فیصل":       "faisal",
    "راولپنڈی":   "rawalpindi",
    "حیدرآباد":   "hyderabad",
    "سیالکوٹ":    "sialkot",
    "گوجرانوالہ": "gujranwala",
    "سندھ":       "sindh",
    "پنجاب":      "punjab",
    "بلوچستان":   "balochistan",
    "خیبر":       "khyber",
    "ہلمند":      "helmand",
    "پختونخوا":   "pakhtunkhwa",
    "افغانستان":  "afghanistan",
    "برطانیہ":    "britain",
    "امریکہ":     "america",
    "ہندوستان":   "hindustan",
    "بھارت":      "bharat",
    "سری":        "sri",
    "لنکا":       "lanka",
    "انگلینڈ":    "england",
    "آسٹریلیا":   "australia",
    "نیوزی":      "newzee",
    "لینڈ":       "land",
    "جنوبی":      "junoobi",
    "افریقہ":     "africa",
    "سعودی":      "saudi",
    "عرب":        "arab",
    "فرانس":      "france",
    "جرمنی":      "germany",
    "ترکی":       "turkey",
    "ایران":      "iran",
    "چین":        "china",
    "روس":        "russia",
    "کابل":       "kabul",
    "ممبئی":      "mumbai",
    "دہلی":       "delhi",
    "بنگلہ":      "bangla",
    "دیش":        "desh",
    "شام":        "sham",
    "عراق":       "iraq",
    "تیونس":      "tunisia",
    "امارات":     "imarat",
    # Organizations
    "بی":         "bi",
    "سی":         "si",
    "تحریک":      "tehreek",
    "انصاف":      "insaf",
    "قومی":       "qaumi",
    "دولتِ":      "daulat-e",
    "دولت":       "daulat",
    "اتحاد":      "ittehad",
    "جماعت":      "jamaat",
    "عوامی":      "awami",
    "متحدہ":      "muttahida",
    "اسلامیہ":    "islamia",
    "کمیشن":      "commission",
    "عدالتی":     "adaalati",
    "داخلہ":      "dakhila",
    # Person names
    "عمران":      "imran",
    "خان":        "khan",
    "نواز":       "nawaz",
    "شریف":       "sharif",
    "بھٹو":       "bhutto",
    "زرداری":     "zardari",
    "مریم":       "maryam",
    "علی":        "ali",
    "محمد":       "muhammad",
    "احمد":       "ahmed",
    "حسین":       "hussain",
    "اسد":        "asad",
    "فواد":       "fawad",
    "چوہدری":     "chaudhry",
    "شاہ":        "shah",
    "ملک":        "malik",
    "سید":        "syed",
    "شیخ":        "sheikh",
    "میاں":       "mian",
    "پرویز":      "pervaiz",
    "آصف":        "asif",
    "منصور":      "mansoor",
    "اعظم":       "azam",
    "حفیظ":       "hafeez",
    "یاسر":       "yasir",
    "محمود":      "mahmood",
    "سرفراز":     "sarfaraz",
    "آفریدی":     "afridi",
    "اللہ":       "allah",
    "الحق":       "ul-haq",
    "مصباح":      "misbah",
    "عادل":       "adil",
    "معین":       "mueen",
    "رشید":       "rashid",
    "الطاف":      "altaf",
    "ذوالفقار":   "zulfikar",
    "جنید":       "junaid",
    "وزیر":       "wazir",
    "وزیرِ":      "wazir-e",
    "وزیراعظم":   "wazir-e-azam",
    "صدر":        "sadar",
    # Numbers / MISC
    "دو":         "do",
    "تین":        "teen",
    "چار":        "chaar",
    "پانچ":       "paanch",
    "سات":        "saat",
    "آٹھ":        "aath",
    "دس":         "das",
    "سو":         "sau",
    "ہزار":       "hazar",
    "لاکھ":       "lakh",
    "کروڑ":       "crore",
    "ایک":        "ek",
    "ارب":        "arab",
    "سنہ":        "sana",
    # Designations
    "جسٹس":       "justice",
    "چیف":        "chief",
    "ہائی":       "high",
    "بریگیڈیئر":  "brigadier",
    "جنرل":       "general",
    "کپتان":      "captain",
    "سپیکر":      "speaker",
    "ڈاکٹر":      "doctor",
}


# ── Two-character aspirated consonants ────
TWO_CHAR_MAP = {
    "بھ": "bh",  "پھ": "ph",  "تھ": "th",
    "ٹھ": "th",  "جھ": "jh",  "چھ": "chh",
    "دھ": "dh",  "ڈھ": "dh",  "کھ": "kh",
    "گھ": "gh",  "لھ": "lh",  "مھ": "mh",
    "نھ": "nh",  "رھ": "rh",
    "آ":  "aa",
    "او": "oo",
    "ای": "ee",
    "اے": "ay",
}

# ── Single character map ──────────────────
SINGLE_CHAR_MAP = {
    "ا": "a",   "آ": "aa",  "ع": "",
    "ء": "",    "ئ": "y",   "ؤ": "w",
    "ب": "b",   "پ": "p",   "ت": "t",
    "ٹ": "t",   "ث": "s",   "ج": "j",
    "چ": "ch",  "ح": "h",   "خ": "kh",
    "د": "d",   "ڈ": "d",   "ذ": "z",
    "ر": "r",   "ڑ": "r",   "ز": "z",
    "ژ": "zh",  "س": "s",   "ش": "sh",
    "ص": "s",   "ض": "z",   "ط": "t",
    "ظ": "z",   "غ": "gh",  "ف": "f",
    "ق": "q",   "ک": "k",   "گ": "g",
    "ل": "l",   "م": "m",   "ن": "n",
    "ں": "n",   "و": "o",   "ہ": "h",
    "ھ": "",    "ی": "i",   "ے": "e",
    # Diacritics
    "\u064E": "a",
    "\u0650": "i",
    "\u064F": "u",
    "\u0651": "",
    "\u0652": "",
    "\u064B": "an",
    "\u064D": "in",
    "\u064C": "un",
    # Digits
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4",
    "۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    # Punctuation
    "۔":".", "،":",", "؟":"?", "؛":";",
}


def map_waw(word, i):
    """و context-aware mapping."""
    if i == 0:
        return "w"
    prev = word[i-1] if i > 0 else ""
    if prev in "اآ":
        return "o"
    return "oo"


def apply_final_vowel(roman, last_urdu_char):
    """Add trailing vowel for words ending in ہ or ا."""
    if not roman:
        return roman
    if roman[-1] in "aeiou":
        return roman
    if last_urdu_char == "ہ":
        return roman + "a"
    if last_urdu_char == "ا":
        return roman + "a"
    return roman


def clean_roman(roman):
    """Collapse 3+ repeated chars, clean spaces."""
    roman = re.sub(r'(.)\1{2,}', r'\1\1', roman)
    roman = re.sub(r'\s+', ' ', roman)
    return roman.strip()


def char_level_transliterate(word):
    """
    Pure character-level transliteration.
    NO vowel insertion — social media Roman Urdu
    naturally drops short vowels between consonants.
    Only diacritics (zabar/zer/pesh) add vowels
    and those are already in SINGLE_CHAR_MAP.
    """
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word):
            two = word[i:i+2]
            if two in TWO_CHAR_MAP:
                result.append(TWO_CHAR_MAP[two])
                i += 2
                continue
        ch = word[i]
        if ch == "و":
            result.append(map_waw(word, i))
        else:
            result.append(SINGLE_CHAR_MAP.get(ch, ch))
        i += 1

    roman = "".join(result)
    roman = apply_final_vowel(roman, word[-1] if word else "")
    roman = clean_roman(roman)
    return roman


def transliterate_word(word):
    """
    Full transliteration pipeline.
    Priority order:
    1. Pass-through: digits, punctuation, Latin
    2. Function word dictionary
    3. English loanword dictionary
    4. Named entity dictionary
    5. Character-level (no vowel insertion)
    """
    if not word or not word.strip():
        return word

    # Pass through digits and pure punctuation
    if re.match(r'^[\d\W]+$', word):
        return word

    # Pass through already-Latin words
    if all(ord(c) < 0x0600 or c in " \t-_'" for c in word):
        return word.lower()

    # Priority 1 — function words (most common)
    if word in FUNCTION_WORDS:
        return FUNCTION_WORDS[word]

    # Priority 2 — English loanwords
    if word in ENGLISH_LOANWORDS:
        return ENGLISH_LOANWORDS[word]

    # Priority 3 — named entities
    if word in NAMED_ENTITY_DICT:
        return NAMED_ENTITY_DICT[word]

    # Priority 4 — character level
    roman = char_level_transliterate(word)
    return roman if roman else word


def transliterate_sentence(token_tag_list):
    return [(transliterate_word(w), t) for w, t in token_tag_list]


def transliterate_dataset(sentences):
    return [transliterate_sentence(s) for s in sentences]


# ── TEST ──────────────────────────────────
if __name__ == "__main__":

    print("── Function Word Test ──\n")
    function_tests = [
        "ہے", "ہیں", "میں", "کو", "سے",
        "نے", "کا", "کی", "کے", "اور",
        "نہیں", "بھی", "پر", "تھا", "تھی",
        "گیا", "ہوا", "کیا", "بہت", "لیکن",
    ]
    for w in function_tests:
        print(f"  {w:<15} → {transliterate_word(w)}")

    print("\n── Entity Word Test ──\n")
    entity_tests = [
        "عمران", "خان", "لاہور", "پاکستان",
        "انگلینڈ", "آسٹریلیا", "تحریک", "انصاف",
        "محمود", "سرفراز", "جسٹس", "پولیس",
    ]
    for w in entity_tests:
        print(f"  {w:<15} → {transliterate_word(w)}")

    print("\n── Full Sentence Test ──\n")
    sample = [
        ("پاکستان", "B-LOC"), ("کی",    "O"),
        ("جانب",    "O"),     ("سے",    "O"),
        ("یاسر",    "B-PER"), ("شاہ",   "I-PER"),
        ("نے",      "O"),     ("لاہور", "B-LOC"),
        ("میں",     "O"),     ("تقریر", "O"),
        ("کی",      "O"),     ("اور",   "O"),
        ("بی بی سی","B-ORG"), ("نے",    "O"),
        ("رپورٹ",   "O"),     ("کی",    "O"),
        ("ہے",      "O"),
    ]
    result = transliterate_sentence(sample)
    print(f"  {'Urdu':<20} {'Tag':<12} {'Roman':<20} Tag")
    print("  " + "─" * 65)
    for (u, t), (r, _) in zip(sample, result):
        print(f"  {u:<20} {t:<12} {r:<20} {t}")

    print("\n── Reading test (paste into browser) ──")
    print("  " + " ".join(r for r, _ in result))