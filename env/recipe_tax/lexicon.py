"""Shared ingredient lexicon: contains-class -> surface words that occur in ingredient lines.
Used by the SFT evidence synthesizer and, since reward v2, by the verifier: a contains claim only counts if some ingredient
line carries a lexicon/trap word for that class. The model's quoted evidence is a hint, not a password."""
from __future__ import annotations

LEX = {  # class -> words likely to appear in an ingredient line (lowercase substrings), checked in order
 "red_meat": ["pepperoni","salami","pig","hamburger","round roast","flank","skirt steak","lamb","mutton","beef","steak","pork","veal","goat","venison","ground meat","bacon","ham","sausage","chorizo","brisket","ribs"],
 "pork": ["pig","pepperoni","salami","spam","lardon","pork","bacon","ham","pancetta","prosciutto","chorizo","lard","salt pork","sausage"],
 "beef": ["round roast","roast beef","hamburger","ground chuck","short rib","flank","skirt steak","tri-tip","corned beef","beef","steak","brisket","oxtail","ground chuck","sirloin","veal"],
 "poultry": ["chicken","turkey","duck","goose","hen","drumstick","thigh","wings"],
 "fish": ["herring","sill","gravlax","lox","smoked salmon","caviar","roe","nam pla","nuoc mam","colatura","fish","salmon","tuna","cod","anchov","sardine","mackerel","haddock","tilapia","trout","fish sauce","dashi","worcestershire","bonito","halibut","snapper","walleye","catfish"],
 "shellfish": ["shrimp","prawn","crab","lobster","clam","mussel","oyster","scallop","squid","calamari","shrimp paste","belacan","oyster sauce","crayfish"],
 "egg": ["miracle whip","custard","egg noodle","frittata","egg","mayonnaise","mayo","aioli","hollandaise","meringue"],
 "dairy": ["pesto","tzatziki","raita","margarine","queso","crema","labneh","kefir","dulce de leche","custard","milk","cream","butter","ghee","cheese","paneer","yogurt","yoghurt","buttermilk","whey","half-and-half","sour cream","ice cream","mascarpone","ricotta","parmesan","mozzarella","cheddar","gruy","feta","curd","khoa","condensed milk","evaporated milk"],
 "honey": ["honey"], "gelatin": ["gelatin","gelatine","marshmallow","jell-o","jello"], "blood": ["blood"],
 "gluten": ["soya","orzo","radiatore","farfalle","rigatoni","fettuccine","vermicelli","matzo","phyllo","filo","puff pastry","pie crust","graham","pretzel","croutons","wonton wrapper","egg roll wrapper","gnocchi","flour","bread","panko","breadcrumb","pasta","spaghetti","noodle","couscous","bulgur","barley","rye","wheat","soy sauce","seitan","semolina","farro","beer","malt","naan","roti","pita","tortilla","cracker","biscuit","cake","muffin","bun","croissant","baguette","dough","crust","oats","worcestershire","udon","ramen","dumpling","wonton","macaroni","penne","linguine","lasagna","bisquick"],
 "tree_nuts": ["gianduja","amaretto","orgeat","almond","cashew","walnut","pistachio","pecan","hazelnut","pine nut","macadamia","brazil nut","chestnut","marzipan","praline","pesto","nutella","frangipane","amaretti","nuts"],
 "peanuts": ["peanut","groundnut","satay"], "soy": ["soya","bean curd","yuba","soy","tofu","tempeh","edamame","miso","gochujang","doenjang","hoisin","tamari","natto","teriyaki"],
 "sesame": ["sesame","tahini","halva","hummus","za'atar","dukkah","benne"], "alcohol": ["wine","beer","mirin","sake","shaoxing","marsala","kirsch","rum","brandy","liqueur","vodka","bourbon","whisk","sherry","cognac","tequila","cider","amaretto","kahlua","vermouth","pisco","champagne"],
 "mustard": ["mustard"], "celery": ["celery","celeriac"],
}

_TRAP_KEY = {"tree_nuts": "nuts"}

def words_for(cls: str, traps: dict) -> list[str]:
    return list(LEX.get(cls, [])) + [str(w).lower() for w in traps.get(_TRAP_KEY.get(cls, cls), [])]

def supporting_line(cls: str, ingredients: list[str], traps: dict, hint: str | None = None) -> str | None:
    """Return the ingredient line that supports `cls`, or None. If `hint` (the model's quoted evidence) is given, lines
    containing the hint are checked first; any line with a lexicon word still counts (the model may quote imprecisely)."""
    ws = words_for(cls, traps)
    lines = [(l, l.lower()) for l in ingredients]
    order = lines
    if hint:
        h = hint.strip().lower()
        if h: order = [x for x in lines if h in x[1]] + [x for x in lines if h not in x[1]]
    for orig, low in order:
        if any(w in low for w in ws): return orig
    return None
