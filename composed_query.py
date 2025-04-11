from itertools import combinations, chain
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
import sentence_transformers
import nltk
from nltk.stem import WordNetLemmatizer
import enchant
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


nltk.download('wordnet') #synonymes
nltk.download('omw-1.4')
nltk.download('punkt') #lemming
nltk.download('stopwords')
stop_words_fr = set(stopwords.words('french'))




spell = SpellChecker(language='fr')
"""1. spell checking"""

def corriger_requete(query):
    mots = query.split()
    mots_corrigés = [spell.correction(mot) or mot for mot in mots]
    return " ".join(mots_corrigés)




"""3. getting similar words"""

fasttext_model = fasttext.load_model('cc.fr.300.bin')

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


"""LFF API request"""

def fetch_and_display_products(query):
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    headers = {
        "Authorization": "Bearer kJgpo2Fu_cmtWpuhATKBh2MZ_VI"
    }
    params = {"text": query}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        products = response.json()
        if products and 'searchPageData' in products and 'results' in products['searchPageData']:
            product_list = products['searchPageData']['results']
            table_data = [[p.get('name', 'N/A')] for p in product_list]
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

    results= [
        (product_names[i], float(cosine_scores[i]))
        for i in range(len(product_names))
        if cosine_scores[i] > threshold
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results



def filtrer_termes(termes_expansion, mots_principaux):
    # Garder les termes qui sont plus proches des mots principaux
    termes_filtres = {terme for terme in termes_expansion if any(mot in terme for mot in mots_principaux)}
    return termes_filtres



def generate_ngrams(query, ngram_range=(1, 3)):
    mots = query.split()
    ngrams = set()
    for n in range(ngram_range[0], ngram_range[1]+1):
        for i in range(len(mots) - n + 1):
            ngram = " ".join(mots[i:i+n])
            ngrams.add(ngram)
    return list(ngrams)




def retirer_stopwords_ngrams(ngrams):
    ngrams_filtres = []
    for ngram in ngrams:
        mots = ngram.split()
        mots_filtres = [mot for mot in mots if mot not in stop_words_fr]
        if mots_filtres:
            ngrams_filtres.append(" ".join(mots_filtres))
    return list(set(ngrams_filtres))  # remove duplicates




def traiter_requete1(query):
    print(f"\n=== Requête initiale : {query}")

    # 1. Correction
    corrected_query = corriger_requete(query.lower())
    print(f"→ Requête corrigée : {corrected_query}")

    # 2. Génération de n-grammes (1 à 3 mots)
    ngrams = generate_ngrams(corrected_query, (1, 3))
    print(f"\n→ N-grammes générés : {ngrams}")

    # 3. Suppression des stop words
    ngrams = retirer_stopwords_ngrams(ngrams)
    print(f"\n→ N-grammes après suppression des stop words : {ngrams}")

    # 4. Lemmatisation
    ngrams = lemming_termes(ngrams)
    print(f"\n→ Termes après lemmatisation : {ngrams}")

   # 4bis. Trier les termes par similarité cosinus avec la requête corrigée
    model_sent = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model_sent.encode([corrected_query])[0]

    terme_sim_scores = []
    for terme in ngrams:
        terme_embedding = model_sent.encode([terme])[0]
        sim_score = cosine_similarity([query_embedding], [terme_embedding])[0][0]
        terme_sim_scores.append((terme, sim_score))

    # Tri décroissant
    termes_tries = [terme for terme, _ in sorted(terme_sim_scores, key=lambda x: x[1], reverse=True)]

    print(f"\n→ Termes triés par similarité cosinus : {termes_tries}")


    # 6. Recherche de produits
    produits_par_terme = {}
    seen_ids = set()

    for terme in termes_tries:
        produits = fetch_and_display_products(terme)
        if produits:
            produits_par_terme[terme] = []
            for p in produits:
                pid = p.get('code')
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    produits_par_terme[terme].append(p)
        else:
            voisins = get_similar_words(terme, k=1)
            for voisin in voisins:
                produits_voisin = fetch_and_display_products(voisin)
                if produits_voisin:
                    print(f"⚠️ Aucun produit pour \"{terme}\". Utilisation du voisin \"{voisin}\".")
                    produits_par_terme[terme] = produits_voisin
                    break

    if not produits_par_terme:
        print("\n❌ Aucun produit trouvé.")
        return

    # 7. Affichage des produits par terme
    print(f"\n=== Résultats détaillés par terme ===")
    for terme, produits in produits_par_terme.items():
        print(f"\n🟢 Terme : \"{terme}\"")
        print("-" * 34)
        for prod in produits:
            print(f"  • {prod.get('name', 'Nom indisponible')}")
        print("-" * 34)

    # 8. Similarité finale avec la requête corrigée
    tous_les_produits = [p for pl in produits_par_terme.values() for p in pl]
    résultats = get_similar_products(tous_les_produits, corrected_query)

    print("\n=== Produits les plus similaires à la requête ===")
    for nom, score in résultats:
        print(f"  🔎 {nom} → {score:.4f}")

traiter_requete1("meuble sous lavabo")