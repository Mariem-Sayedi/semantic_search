from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from pipeline_semantic_search import corriger_requete, synonym_query, get_similar_words, fetch_and_display_products, get_similar_products, normaliser_termes, filtrer_termes
import re


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
        # synonymes
        synonymes = synonym_query(mot)
        synonymes_corrigés = {corriger_requete(s).lower() for s in synonymes}
        termes_expansion.update(synonymes_corrigés)

        # voisins fastText
        similaires = get_similar_words(mot)
        similaires_corrigés = {corriger_requete(s).lower() for s in similaires}
        termes_expansion.update(similaires_corrigés)

    print(f"\ntermes d’expansion avant lemming: {termes_expansion}")
    termes_expansion = normaliser_termes(termes_expansion)
    print(f"\ntermes d’expansion après lemming: {termes_expansion}")

    # Filtrage des termes pour ne garder que les plus pertinents
    termes_expansion = filtrer_termes(termes_expansion, mots_principaux)
    termes_expansion = normaliser_termes(termes_expansion)
    print(f"\ntermes d’expansion après filtrage : {termes_expansion}")

    produits_par_terme = {}
    seen_ids = set()

    for terme in termes_expansion:
        produits = fetch_and_display_products(terme)
        if produits:
            produits_par_terme[terme] = []
            for prod in produits:
                prod_id = prod.get('code')
                if prod_id and prod_id not in seen_ids:
                    seen_ids.add(prod_id)
                    produits_par_terme[terme].append(prod)
        else:
            # Si aucun produit n'est trouvé, on cherche parmi les voisins fastText
            voisins = get_similar_words(terme)
            for voisin in voisins:
                produits_voisin = fetch_and_display_products(voisin)
                if produits_voisin:  # Si le voisin donne des produits, on l'utilise
                    print(f"Pas de produits pour le terme \"{terme}\". Utilisation du voisin \"{voisin}\".")
                    produits_par_terme[terme] = produits_voisin
                    break

    if not produits_par_terme:
        print("Aucun produit trouvé.")
        return

    print(f"\nrésultats détaillés par terme :")
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
    return {
        "requete_corrigee": corrected_query,
        "termes_expansion": list(termes_expansion),
        "resultats": produits_par_terme
    }
