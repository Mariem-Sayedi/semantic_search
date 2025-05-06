import fasttext
from spellchecker import SpellChecker
from nltk.corpus import wordnet as wn
import nltk
import re
import requests
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import enchant
from sentence_transformers import SentenceTransformer


nltk.download('wordnet') #synonymes
nltk.download('omw-1.4')
nltk.download('punkt') #lemming

spell = SpellChecker(language='fr')
model_sent = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


"""1. spell checking"""

def corriger_requete(query):
    mots = query.split()
    mots_corrigés = [spell.correction(mot) or mot for mot in mots]
    return " ".join(mots_corrigés)


"""3. getting similar words"""

fasttext_model = fasttext.load_model('semantic_search/cc.fr.300.bin')

def get_similar_words(word, k=5):
    try:
        if word not in fasttext_model.words:
            return []
        similar = fasttext_model.get_nearest_neighbors(word, k)
        return [w for _, w in similar]
    except Exception as e:
        print(f"Erreur avec fastText pour le mot '{word}': {e}")
        return []


"""Lemming"""


lemmatizer = WordNetLemmatizer()

def lemming_termes(termes):
    termes_lemmés = set()
    for terme in termes:
        lemma = lemmatizer.lemmatize(terme)
        termes_lemmés.add(lemma)
    return termes_lemmés


dico_fr = enchant.Dict("fr_FR")

def filtrer_mots_francais(termes):
    return {mot for mot in termes if dico_fr.check(mot)}



import requests

# Création d'une session persistante
session = requests.Session()

def get_access_token():
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/token"

    response = session.get(url)

    print("Status:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        return data.get("access_token")
    return None




"""LFF API request"""

def fetch_and_display_products(query, store_id):
    access_token = get_access_token()

    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
    }
    cookies = {"preferredStoreCode": str(store_id)}

    params = {"text": query}
    response = requests.get(url, headers=headers, params=params, cookies=cookies)

    if response.status_code == 200:
        products = response.json()
        if products and 'searchPageData' in products and 'results' in products['searchPageData']:
            product_list = products['searchPageData']['results']
            table_data = [[p.get('name', 'code')] for p in product_list]
            if table_data:
              return product_list
    print("erreur ou aucun produit trouvé.")
    return []

"""4. Semantic similarity between products and query"""



def get_similar_products(product_list, query, threshold=0.5):
    product_names = [product.get('name', '') for product in product_list]
    if not product_names:
        return []

    model_sent = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    product_embeddings = model_sent.encode(product_names, convert_to_tensor=True)
    query_embedding = model_sent.encode(query, convert_to_tensor=True).reshape(1, -1)
    product_embeddings = product_embeddings.reshape(len(product_embeddings), -1)

    cosine_scores = cosine_similarity(query_embedding, product_embeddings)[0]

    results = [
        (product_list[i], float(cosine_scores[i]))
        for i in range(len(product_list))
        if cosine_scores[i] > threshold
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def filtrer_termes(termes_expansion, mots_principaux):
    # garder les termes qui sont plus proches de la requête principale
    termes_filtres = {terme for terme in termes_expansion if any(mot in terme for mot in mots_principaux)}
    return termes_filtres

"""5. Complete pipeline"""


def traiter_requete(query, store_id):
    
    # Correction orthographique
    corrected_query = corriger_requete(query)

    # Extraction des mots principaux
    mots = re.findall(r'\w+', corrected_query.lower())

    # Expansion : fastText + correction
    termes_expansion = set(mots)
    for mot in mots:
        similaires = get_similar_words(mot)
        similaires_corrigés = {corriger_requete(s).lower() for s in similaires}
        termes_expansion.update(similaires_corrigés)

    # Lemmatisation
    termes_expansion = lemming_termes(termes_expansion)

    # Filtrage par dictionnaire français
    termes_expansion = filtrer_mots_francais(termes_expansion)

    # Tri des termes par similarité cosinus avec la requête
    query_embedding = model_sent.encode([corrected_query])[0]

    terme_sim_scores = []
    for terme in termes_expansion:
        terme_embedding = model_sent.encode([terme])[0]
        sim_score = cosine_similarity([query_embedding], [terme_embedding])[0][0]
        terme_sim_scores.append((terme, sim_score))

    # Tri décroissant
    termes_tries = [terme for terme, _ in sorted(terme_sim_scores, key=lambda x: x[1], reverse=True)]

    # Appel API uniquement avec la requête corrigée
    produits = fetch_and_display_products(corrected_query, store_id)

    if produits:
      résultats = get_similar_products(produits, corrected_query)
      response = [
        {
            "product_name": product.get('name', ''),
            "product_code": product.get('code', ''),
            "cosine_similarity": round(score, 3)
        }
        for product, score in résultats
    ]
      return {
        "query_corrected": corrected_query,
        "expanded_terms": termes_tries,
        "results": response
    }
    else:
      return {
        "query_corrected": corrected_query,
        "expanded_terms": termes_tries,
        "results": []
    }



