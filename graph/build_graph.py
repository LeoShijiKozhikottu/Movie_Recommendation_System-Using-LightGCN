import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(ROOT_DIR, "data", "ratings_with_titles.csv")
df = pd.read_csv(csv_path)

# Encode user and item IDs
user_enc = LabelEncoder()
item_enc = LabelEncoder()

user_ids = user_enc.fit_transform(df["user"])
item_ids = item_enc.fit_transform(df["item"])

num_users = int(len(user_enc.classes_))
num_items = int(len(item_enc.classes_))

# Shift item IDs into bipartite graph index space
shifted_items = item_ids + num_users

# Construct bidirectional edges [u, i] and [i, u]
u = torch.as_tensor(user_ids, dtype=torch.long)
i = torch.as_tensor(shifted_items, dtype=torch.long)

src = torch.empty(2 * len(u), dtype=torch.long)
dst = torch.empty(2 * len(i), dtype=torch.long)
src[0::2] = u
src[1::2] = i
dst[0::2] = i
dst[1::2] = u

edge_index = torch.stack([src, dst], dim=0)

data = Data(edge_index=edge_index)

# Save graph package
save_path = os.path.join(ROOT_DIR, "graph", "graph_data.pt")
torch.save({
    "graph": data,
    "num_users": num_users,
    "num_items": num_items
}, save_path)

print("Graph rebuilt successfully!")
