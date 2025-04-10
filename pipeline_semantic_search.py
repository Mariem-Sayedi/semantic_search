# -*- coding: utf-8 -*-
"""pipeline_semantic_search.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wO4N2IGOAWUIzXcuj-TQAYSVg7THPmEA
"""

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

nltk.download('wordnet') #synonymes
nltk.download('omw-1.4')
nltk.download('punkt') #stemming, lemming

"""1. spell checking"""

def corriger_requete(query, langue='fr'):
    spell = SpellChecker(language=langue)
    mots = query.split()
    mots_corrigés = [spell.correction(mot) or mot for mot in mots]
    return " ".join(mots_corrigés)

"""2. getting synonyms"""

def synonym_query(query):
    words = query.split()
    expanded_words = set(words)
    for word in words:
        synonyms = wn.synsets(word, lang='fra')
        for syn in synonyms:
            for lemma in syn.lemmas('fra'):
                expanded_words.add(lemma.name())
    return expanded_words



"""3. getting similar words"""

fasttext_model = fasttext.load_model('cc.fr.300.bin')

def get_similar_words(word, k=10):
    try:
        similar = fasttext_model.get_nearest_neighbors(word, k)
        return [w for _, w in similar]
    except Exception as e:
        print(f"Erreur avec fastText pour le mot '{word}': {e}")
        return []

"""Lemming"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def normaliser_termes(termes):
    termes_normalisés = set()
    for terme in termes:
        lemma = lemmatizer.lemmatize(terme)
        termes_normalisés.add(lemma)
    return termes_normalisés

"""LFF API request"""

def fetch_and_display_products(query):
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    headers = {
        "Authorization": "Bearer MsxJVignZomfbtw_Ed6HHJDoCiI"
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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_products(product_list, query, threshold=0.3):
    product_names = [product.get('name', '') for product in product_list]
    if not product_names:
        return []

    model_sent = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    product_embeddings = model_sent.encode(product_names, convert_to_tensor=True)
    query_embedding = model_sent.encode(query, convert_to_tensor=True).reshape(1, -1)
    product_embeddings = product_embeddings.reshape(len(product_embeddings), -1)

    cosine_scores = cosine_similarity(query_embedding, product_embeddings)[0]

    return [
        (product_names[i], float(cosine_scores[i]))
        for i in range(len(product_names))
        if cosine_scores[i] > threshold
    ]

"""5. Complete pipeline"""

def traiter_requete(query):
    print(f"\n requête initiale : {query}")
    corrected_query = corriger_requete(query)
    print(f"requête corrigée : {corrected_query}")

    mots = re.findall(r'\w+', corrected_query.lower())
    termes_expansion = set(mots)

    for mot in mots:
        # synonymes
        synonymes = synonym_query(mot)
        synonymes_corrigés = {corriger_requete(s).lower() for s in synonymes}
        termes_expansion.update(synonymes_corrigés)

        # voisins fastText
        similaires = get_similar_words(mot)
        similaires_corrigés = {corriger_requete(s).lower() for s in similaires}
        termes_expansion.update(similaires_corrigés)


    print(f"\n termes d’expansion avant lemming: {termes_expansion}")
    termes_expansion = normaliser_termes(termes_expansion)
    print(f"\n termes d’expansion après lemming: {termes_expansion}")


    produits_par_terme = {}
    seen_ids = set()

    for terme in termes_expansion:
        produits = fetch_and_display_products(terme)
        if not produits:
            continue
        produits_par_terme[terme] = []
        for prod in produits:
            prod_id = prod.get('code')
            if prod_id and prod_id not in seen_ids:
                seen_ids.add(prod_id)
                produits_par_terme[terme].append(prod)

    if not produits_par_terme:
        print(" aucun produit trouvé.")
        return

    print(f"\n résultats détaillés par terme :")
    for terme, produits in produits_par_terme.items():
        print(f"\n terme ➜ \"{terme}\"\n" + "-" * 34)
        for prod in produits:
            nom = prod.get("name", "Nom indisponible")
            print(f"  produit : {nom}\n ")
        print("-" * 34)

    tous_les_produits = [p for prods in produits_par_terme.values() for p in prods]
    résultats = get_similar_products(tous_les_produits, corrected_query)

    print("\n produits similaires à la requête :")
    for nom, score in résultats:
        print(f"  {nom} --> {score:.4f}")

traiter_requete("téléphoen")

