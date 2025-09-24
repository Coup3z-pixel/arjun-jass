import pandas as pd
import re
import numpy as np

class DictatorGameIndexer:
    def __init__(self, csv_file):
        """
        :param csv_file: path to CSV

        CSV columns:
        llm_name,response,scenario_type,endowment,num_recipients,
        work_contribution,project_context,team_relationship,prompt
        """
        self.csv_file = csv_file

        self.df = None
        self.llm_to_index = {}
        self.index_to_llm = {}
        self.altruism = {}

        self._load_data()
        self._build_index()
        self._compute_measures()

    def _load_data(self):
        """Load CSV into pandas and parse Keep/Donate values from response."""
        df = pd.read_csv(self.csv_file, engine="python", quoting=1, on_bad_lines="skip")

        # Clean numeric fields
        df["endowment"] = pd.to_numeric(df["endowment"], errors="coerce")
        df["num_recipients"] = pd.to_numeric(df.get("num_recipients", 1), errors="coerce").fillna(1).astype(int)

        # Extract Keep and Donate as percentages
        df["keep_percent"] = df["response"].str.extract(r'Keep\s*(\d+)%').astype(float)
        df["donate_percent"] = df["response"].str.extract(r'Donate\s*(\d+)%').astype(float)

        # Drop rows where percentages missing
        df = df.dropna(subset=["keep_percent", "donate_percent", "endowment"])

        # Convert percentages into absolute amounts
        df["keep"] = (df["keep_percent"] / 100.0) * df["endowment"]
        df["donate"] = (df["donate_percent"] / 100.0) * df["endowment"]

        self.df = df

    def _build_index(self):
        """Build mappings between LLM names and indices."""
        unique_llms = self.df["llm_name"].str.strip().unique()
        self.llm_to_index = {llm: i for i, llm in enumerate(unique_llms)}
        self.index_to_llm = {i: llm for llm, i in self.llm_to_index.items()}

    def _compute_measures(self):
        """
        Compute altruism indexes:
        α = (U_D - keep) / donate
        β = (keep - U_D) / (keep - donate)
        θ = (U_D - keep) / ln(1 + donate)

        Assumption:
        U_D = keep + donate  (dictator values both own and given payoff equally)
        """
        df = self.df.copy()

        # Dictator utility assumption (can be adjusted later if needed)
        df["UD"] = df["keep"] + df["donate"]

        # α: Altruism parameter
        df["alpha"] = (df["UD"] - df["keep"]) / df["donate"]

        # β: Inequity aversion parameter (avoid div by 0)
        df["beta"] = np.where(
            (df["keep"] - df["donate"]) != 0,
            (df["keep"] - df["UD"]) / (df["keep"] - df["donate"]),
            0
        )

        # θ: Intrinsic giving satisfaction (avoid log(0))
        df["theta"] = np.where(
            df["donate"] > 0,
            (df["UD"] - df["keep"]) / np.log1p(df["donate"]),
            np.nan
        )

        # Aggregate averages per LLM into a nested dict
        grouped = df.groupby("llm_name").agg({
            "alpha": "mean",
            "beta": "mean",
            "theta": "mean",
            "UD": "mean"
        })

        self.altruism = grouped.to_dict(orient="index")
        self.df = df  # keep enriched dataframe

    # -------------------
    # Access Methods
    # -------------------
    def get_index(self, llm):
        return self.llm_to_index.get(llm)

    def get_llm(self, index):
        return self.index_to_llm.get(index)

    def all_indices(self):
        return self.llm_to_index

    def get_altruism(self, llm):
        """Return dict of {alpha, beta, theta, UD} for a given LLM"""
        return self.altruism.get(llm, {})
