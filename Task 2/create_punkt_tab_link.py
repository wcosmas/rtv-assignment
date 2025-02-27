# create_punkt_tab_link.py
# Create a symbolic link from punkt to punkt_tab

import os
import nltk
from nltk.data import path as nltk_data_path

# Get the NLTK data directory
nltk_dir = nltk_data_path[0]

# Source directory (punkt)
punkt_dir = os.path.join(nltk_dir, 'tokenizers', 'punkt')

# Target directory (punkt_tab)
punkt_tab_dir = os.path.join(nltk_dir, 'tokenizers', 'punkt_tab')

# Create the punkt_tab directory if it doesn't exist
os.makedirs(punkt_tab_dir, exist_ok=True)

# Create a symbolic link from punkt/english to punkt_tab/english
source = os.path.join(punkt_dir, 'english')
target = os.path.join(punkt_tab_dir, 'english')

if os.path.exists(source) and not os.path.exists(target):
    try:
        os.symlink(source, target)
        print(f"Created symbolic link from {source} to {target}")
    except Exception as e:
        print(f"Error creating symbolic link: {e}")
else:
    print(f"Source {source} doesn't exist or target {target} already exists") 