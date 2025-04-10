from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from pipeline_semantic_search import corriger_requete, synonym_query, get_similar_words, fetch_and_display_products, get_similar_products, normaliser_termes
import re


app = FastAPI()

# Exemple d'entrée (requête utilisateur)
class QueryModel(BaseModel):
    query: str

@app.post("/search/")
def traiter_recherche(input: QueryModel):
    query = input.query

    
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

    return {
        "requete_corrigee": corrected_query,
        "termes_expansion": list(termes_expansion),
        "resultats": produits_par_terme
    }
