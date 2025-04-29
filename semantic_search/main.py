from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from semantic_search.pipeline_semantic_search import corriger_requete, get_similar_words, fetch_and_display_products, get_similar_products, lemming_termes, filtrer_mots_francais
import re
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


app = FastAPI()

# Exemple d'entrée (requête utilisateur)
class QueryModel(BaseModel):
    query: str





@app.post("/search/")
def traiter_recherche(input: QueryModel):
    print(f"\nrequête initiale : {input.query}")
    corrected_query = corriger_requete(input.query)
    print(f"requête corrigée : {corrected_query}")

    mots = re.findall(r'\w+', corrected_query.lower())
    mots_principaux = set(mots)

    termes_expansion = set(mots)
    
    for mot in mots:

        # voisins fastText
        similaires = get_similar_words(mot)
        similaires_corrigés = {corriger_requete(s).lower() for s in similaires}
        termes_expansion.update(similaires_corrigés)

    print(f"\ntermes d’expansion avant lemming: {termes_expansion}")
    termes_expansion = lemming_termes(termes_expansion)
    print(f"\ntermes d’expansion après lemming: {termes_expansion}")

    print(f"\ntermes d’expansion avant vérification: {termes_expansion}")
    termes_expansion = filtrer_mots_francais(termes_expansion)
    print(f"\ntermes d’expansion après vérification: {termes_expansion}")

    # 4bis. Trier les termes par similarité cosinus avec la requête corrigée
    model_sent = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model_sent.encode([corrected_query])[0]

    terme_sim_scores = []
    for terme in termes_expansion:
        terme_embedding = model_sent.encode([terme])[0]
        sim_score = cosine_similarity([query_embedding], [terme_embedding])[0][0]
        terme_sim_scores.append((terme, sim_score))

    # Tri décroissant
    termes_tries = [terme for terme, _ in sorted(terme_sim_scores, key=lambda x: x[1], reverse=True)]

    print(f"\n Termes triés par similarité cosinus : {termes_tries}")


    produits_par_terme = {}
    seen_ids = set()

    for terme in termes_tries:
        produits = fetch_and_display_products(terme)
        if produits:
            produits_par_terme[terme] = []
            for prod in produits:
                prod_id = prod.get('code')
                if prod_id and prod_id not in seen_ids:
                    seen_ids.add(prod_id)
                    produits_par_terme[terme].append(prod)
        else:
           
            voisins = get_similar_words(terme, k=1)  
            for voisin in voisins:
                produits_voisin = fetch_and_display_products(voisin)
                if produits_voisin:  
                    print(f"Pas de produits pour le terme \"{terme}\". Utilisation du voisin \"{voisin}\".")
                    produits_par_terme[terme] = produits_voisin
                    break

    if not produits_par_terme:
        print("Aucun produit trouvé.")
        return

    print(f"\n résultats détaillés par terme :")
    for terme, produits in produits_par_terme.items():
        print(f"\nterme ➜ \"{terme}\"\n" + "-" * 34)
        for prod in produits:
            nom = prod.get("name", "Nom indisponible")
            print(f"  produit : {nom}\n ")
        print("-" * 34)

    tous_les_produits = [p for prods in produits_par_terme.values() for p in prods]
    résultats = get_similar_products(tous_les_produits, corrected_query)

    print("\nproduits similaires à la requête :")
    for nom, score in résultats:
        print(f"  {nom} --> {score:.4f}")

    print("\nproduits similaires à la requête :")
    for nom, score in résultats:
        print(f"  {nom} --> {score:.4f}")

    return {
        "requete_corrigee": corrected_query,
        "termes_expansion": list(termes_expansion),
        "resultats": produits_par_terme
    }
