import pandas as pd



def score_navigation_client(product_id, user):
    score = 0
    if product_id in user.browsed_categories:
        score += 1
    if product_id in user.viewed_products:
        score += 3
    if product_id in user.cart:
        score += 5
    if product_id in user.purchased_products:
        score += 10
    return score



def score_collaboratif(product_id: str, user_guid: str, matrix: pd.DataFrame, similarity_df: pd.DataFrame) -> float:
    if user_guid not in similarity_df.index or product_id not in matrix.columns:
        return 0.0

    similar_users = similarity_df[user_guid].drop(index=user_guid)
    score = 0.0
    total_sim = 0.0

    for other_user, sim in similar_users.items():
        if matrix.at[other_user, product_id] == 1:
            score += sim
            total_sim += sim

    return score / total_sim if total_sim > 0 else 0.0

def score_localisation(product, store):
    return 1.0 if store in product.available_stores else 0.0



def score_tendance_marche(product):
    return 0.7 * product.nb_ventes + 0.3 * product.nb_consultations


def score_promotion(product):
    return 1.0 if product.promo_active else 0.3


def score_saison(product, date):
    return 1.0 if product.is_seasonal(date) else 0.3


def score_boost_admin(product, date):
    if product.boost_start <= date <= product.boost_end:
        return product.boost_value
    return 0.0


def score_total(product, user, context, date, similar_users):
    return (
        0.25 * score_navigation_client(product.id, user)
      + 0.20 * score_collaboratif(product.id, user, similar_users)
      + 0.10 * score_localisation(product, context["store"])
      + 0.10 * score_tendance_marche(product)
      + 0.10 * score_promotion(product)
      + 0.10 * score_saison(product, date)
      + 0.15 * score_boost_admin(product, date)
    )


def score_final(score_semantique, score_total, importance_semantique=0.7):
    importance_total = 1.0 - importance_semantique
    return importance_semantique * score_semantique + importance_total * score_total
