import fasttext
import re
import nltk
import stanza
import enchant
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from custom_search_ranking.app.services.search_products_api import fetch_all_products
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from nltk.stem import WordNetLemmatizer



# Setup unique
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
stanza.download('fr', processors='tokenize,pos,lemma', verbose=False)

# Initialisation une fois
spell = SpellChecker(language='fr')
dico_fr = enchant.Dict("fr_FR")
nlp_fr = stanza.Pipeline(lang='fr', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)

# Chargement stopwords
with open("semantic_search/stopwords_fr.txt", "r", encoding="utf-8") as f:
    stopwords_fr = set(line.strip().lower() for line in f if line.strip())

@lru_cache(maxsize=1)
def get_fasttext_model():
    return fasttext.load_model('semantic_search/cc.fr.300.bin')

@lru_cache(maxsize=1)
def get_sentence_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def corriger_requete(query: str) -> str:
    return " ".join(spell.correction(mot) or mot for mot in query.split())

def get_similar_words(word: str, k=7, threshold=0.5):
    try:
        model = get_fasttext_model()
        if word not in model.words:
            return []
        voisins = model.get_nearest_neighbors(word)
        return [w for score, w in voisins if score >= threshold and w != word][:k]
    except Exception:
        return []

def filtrer_mots_francais(termes):
    return {mot for mot in termes if dico_fr.check(mot)}

def process_with_stanza(termes):
    noms = set()
    doc = nlp_fr(' '.join(termes))
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos != 'VERB':
                noms.add(word.text.lower())
    return noms


lemmatizer = WordNetLemmatizer()

def lemming_termes(termes):
    termes_lemmés = set()
    for terme in termes:
        lemma = lemmatizer.lemmatize(terme)
        termes_lemmés.add(lemma)
    return termes_lemmés




def traiter_requete(query, store_id):
    query = query.lower()
    corrected_query = corriger_requete(query)
    mots = re.findall(r'\w+', corrected_query)

    # Expansion des termes + correction dans un threadpool
    def expand(mot):
        similaires = get_similar_words(mot)
        return set(similaires)

    termes_expansion = set(mots)
    with ThreadPoolExecutor() as executor:
        for result in executor.map(expand, mots):
            termes_expansion.update(result)

    # NLP + Lemmatisation + noms
    termes_noms = process_with_stanza(termes_expansion)
    termes_noms = filtrer_mots_francais(termes_noms)

    termes_noms = lemming_termes(termes_noms)

    if not termes_noms:
        return {"query_corrected": corrected_query, "expanded_terms": [], "results": []}

    # Similarité sémantique
    model_sent = get_sentence_model()
    termes_list = list(termes_noms)
    embeddings = model_sent.encode(termes_list + [corrected_query], convert_to_tensor=False)
    query_embedding = embeddings[-1]
    termes_embeddings = embeddings[:-1]

    sim_scores = cosine_similarity([query_embedding], termes_embeddings)[0]
    termes_tries = [terme for terme, _ in sorted(zip(termes_list, sim_scores), key=lambda x: x[1], reverse=True)]

    # Récupération produits
    produits_df, pagination = fetch_all_products(corrected_query, store_id)

    if produits_df.empty:
        return {"query_corrected": corrected_query, "expanded_terms": termes_tries, "results": []}

    product_names = produits_df["product_name"].tolist()
    all_embeddings = model_sent.encode(product_names + [corrected_query], convert_to_tensor=False)
    product_embeddings = all_embeddings[:-1]
    query_embedding = all_embeddings[-1]

    scores = cosine_similarity([query_embedding], product_embeddings)[0]
    produits_df["semantic_similarity"] = scores
    produits_df = produits_df[produits_df["semantic_similarity"] > 0.3]
    produits_df = produits_df.sort_values(by="semantic_similarity", ascending=False)

    results = produits_df.apply(lambda row: {
        "product_name": row["product_name"],
        "product_id": row["product_id"],
        "product_url": row["product_url"],
        "promo_rate": row["promo_rate"],
        "product_brand": row["product_brand"],
        "image_url": row["image_url"],
        "price": row["price"],
        "gross_price": row["gross_price"],
        "store_stock_price": row["store_stock_price"],
        "semantic_similarity": round(row["semantic_similarity"], 3)
    }, axis=1).tolist()

    return {
        "query_corrected": corrected_query,
        "expanded_terms": termes_tries,
        "results": results,
        "pagination": pagination
    }
