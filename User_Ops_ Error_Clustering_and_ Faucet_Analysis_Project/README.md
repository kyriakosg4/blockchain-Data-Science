# Error Clustering & Faucet Analysis on Account Abstraction

This project extracts and analyzes data from an **Account Abstraction (AA)** testnet.  
It includes:

- Fetching **block-level** transactions
- Decoding **UserOperation events**
- **Error clustering** at the UserOperation level
- Visualizing results
- Analyzing **faucet clicks** and detecting potential **bot accounts**

---

## 📂 Project Structure

Error_Clustering_Data_Analysis_Project/
│
├── blocks/ # Raw block & transaction CSVs
├── data/ # Processed CSVs (summaries, faucet clicks, wallet behavior)
├── faucet_visuals/ # Faucet visualizations
├── logic/ # Core Python logic scripts
│ ├── fetch_blocks.py
│ ├── decode_userops.py
│ ├── cluster_errors.py
│ ├── visualize.py
│ ├── faucet_click_analysis.py
│ ├── faucet_visualize.py
│ └── wallet_behavior.py
├── uops/ # Decoded UserOperation CSVs
├── visualizations/ # Error clustering & userop failure plots
├── main.ipynb # Main Jupyter Notebook runner
├── requirements.txt # Dependencies
├── .env.example # Example environment variables
├── .gitignore # Files to exclude from Git
└── README.md # Project description

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/blockchain-Data-Science.git
   cd blockchain-Data-Science/Error_Clustering_Data_Analysis_Project


2. Create a virtual environment:
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate


3. Install dependencies:
  pip install -r requirements.txt


4. requirements.txt

  # Core
  pandas
  numpy
  tqdm
  python-dotenv
  
  # Async + networking
  nest_asyncio
  aiohttp
  web3
  
  # ABI decoding
  eth-abi
  
  # Visualization`
  matplotlib
  seaborn
  plotly
  
  # Jupyter
  notebook
  jupyterlab
 



