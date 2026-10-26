"""Country facts for two-hop prompts (capital -> country -> language) and their one-hop controls."""

# (country, capital, official language); countries with one clearly dominant official language
COUNTRIES = [
    ("France", "Paris", "French"), ("Germany", "Berlin", "German"), ("Italy", "Rome", "Italian"),
    ("Spain", "Madrid", "Spanish"), ("Portugal", "Lisbon", "Portuguese"), ("Greece", "Athens", "Greek"),
    ("Poland", "Warsaw", "Polish"), ("Netherlands", "Amsterdam", "Dutch"), ("Sweden", "Stockholm", "Swedish"),
    ("Norway", "Oslo", "Norwegian"), ("Finland", "Helsinki", "Finnish"), ("Hungary", "Budapest", "Hungarian"),
    ("Denmark", "Copenhagen", "Danish"), ("Romania", "Bucharest", "Romanian"), ("Russia", "Moscow", "Russian"),
    ("Ukraine", "Kyiv", "Ukrainian"), ("Bulgaria", "Sofia", "Bulgarian"), ("Serbia", "Belgrade", "Serbian"),
    ("Croatia", "Zagreb", "Croatian"), ("Austria", "Vienna", "German"), ("Turkey", "Ankara", "Turkish"),
    ("Japan", "Tokyo", "Japanese"), ("China", "Beijing", "Chinese"), ("Thailand", "Bangkok", "Thai"),
    ("Vietnam", "Hanoi", "Vietnamese"), ("Indonesia", "Jakarta", "Indonesian"), ("Iran", "Tehran", "Persian"),
    ("Israel", "Jerusalem", "Hebrew"), ("Pakistan", "Islamabad", "Urdu"), ("Malaysia", "Kuala Lumpur", "Malay"),
    ("Egypt", "Cairo", "Arabic"), ("Morocco", "Rabat", "Arabic"), ("Ethiopia", "Addis Ababa", "Amharic"),
    ("Kenya", "Nairobi", "Swahili"), ("Brazil", "Brasília", "Portuguese"), ("Argentina", "Buenos Aires", "Spanish"),
    ("Peru", "Lima", "Spanish"), ("Chile", "Santiago", "Spanish"), ("Colombia", "Bogotá", "Spanish"),
    ("Venezuela", "Caracas", "Spanish"), ("Cuba", "Havana", "Spanish"), ("Australia", "Canberra", "English"),
    ("Nigeria", "Abuja", "English"),
]

PREFIX = "Here is a question about world geography, answered in one word."
# Several phrasings, so that per-country averages can be taken over prompts other than the one patched
TWO_HOP = [
    PREFIX + " The official language of the country whose capital is {capital} is",
    PREFIX + " In the country whose capital city is {capital}, the official language is",
    PREFIX + " The main language spoken in the nation with the capital {capital} is",
    PREFIX + " People in the country that has {capital} as its capital mostly speak",
]
ONE_HOP = [
    PREFIX + " The official language of {country} is",
    PREFIX + " In {country}, the official language is",
    PREFIX + " The main language spoken in {country} is",
    PREFIX + " People in {country} mostly speak",
]
