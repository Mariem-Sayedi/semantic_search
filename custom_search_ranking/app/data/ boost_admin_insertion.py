import sqlite3
import random
from datetime import datetime, timedelta



DB_PATH = "custom_search_ranking/app/data/LFF.db"

# Connexion à la base SQLite (ou création si elle n'existe pas)
conn = sqlite3.connect(DB_PATH)
               
cursor = conn.cursor()



# Step 1: Create a new table with the correct column types
# cursor.execute('''
#   CREATE TABLE admin_boosts (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     target_type TEXT,
#     target_id INTEGER,
#     store_id TEXT,
#     boost_score REAL,
#     start_date DATETIME,
#     end_date DATETIME
# );

# ''')


# 3. Liste unique de produits
product_ids = list(set([
    10000062727, 10000171279, 10000192981, 100000533674, 10000210146, 10000186608,
    10000242202, 10000259553, 10000005253, 100000516034, 10000382695, 10000214682,
    100000517283, 10000254806, 10000211448, 10000240536, 100000530726, 10000270069,
    10000224237, 10000266786, 100000375264, 10000223844, 10000224074, 10000254923,
    10000169763, 10000248615, 100000349424, 10000259554, 10000253090, 10000162002,
    10000234582, 10000224104, 100000504310, 100000363938, 10000253163, 10000257729,
    100000375570, 10000242655, 10000231094, 10000159038, 10000266775, 10000136691,
    10000186994, 10000218736, 10000201738, 10000104430, 10000208654, 100000520114,
    10000254851, 10000278526, 10000173887, 10000149901, 100000365691, 10000255923,
    10000253467, 10000262357, 10000168187, 10000215753, 10000262453, 10000261091,
    100000504182, 10000201745
]))

today = datetime.today()

# 4. Insertion des données aléatoires
for pid in product_ids:
    start_date = today - timedelta(days=random.randint(0, 30))
    end_date = start_date + timedelta(days=random.randint(7, 30))
    boost_score = round(random.uniform(1, 5), 2)

    cursor.execute('''
        INSERT INTO admin_boosts (target_type, target_id, store_id, boost_score, start_date, end_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        'Produit',
        pid,
        '0008',
        boost_score,
        start_date,
        end_date
    ))

# 5. Commit & close
conn.commit()
conn.close()

print(" Table admin_boosts créée et alimentée avec des données aléatoires.")