import pandas as pd
from collections import defaultdict

class NonAtomicIndexer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.llm_to_index = {}
        self.index_to_llm = {}
        self.altruism = {}
        self._load_data()
        self._build_index()
        self._derive_altruism()

    def _load_data(self):
        """Load CSV into pandas, clean data types, and drop invalid rows."""
        self.df = pd.read_csv(
            self.csv_file,
            engine="python",
            quoting=1,
            on_bad_lines="skip"
        )
        # Drop rows without LLM or round info
        self.df = self.df.dropna(subset=["llm", "round"])
        # Strip whitespace from strings
        self.df["llm"] = self.df["llm"].str.strip()
        # Convert numeric fields safely
        numeric_cols = ["round", "consumption", "fish_num", "fishermen_num"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df = self.df.dropna(subset=numeric_cols)

    def _build_index(self):
        """Build mappings between LLM names and indices."""
        unique_llms = self.df["llm"].unique()
        self.llm_to_index = {llm: i for i, llm in enumerate(unique_llms)}
        self.index_to_llm = {i: llm for llm, i in self.llm_to_index.items()}

    def _derive_altruism(self):
        """Compute average altruism measures for each LLM across all rounds."""
        df = self.df.copy()
        altruism_data = defaultdict(lambda: defaultdict(list))

        # Compute per-round measures
        for round_num, group in df.groupby("round"):
            Xmax = group["consumption"].max()
            impacts = group["consumption"] / group["fish_num"]
            max_impact = impacts.max()
            selfish_payoff = group.iloc[0]["fish_num"] / group.iloc[0]["fishermen_num"]

            for idx, row in group.iterrows():
                llm = row["llm"]
                Xi = row["consumption"]
                # Measures
                rel_harvest = (Xi / Xmax)
                marginal_impact = (impacts[idx] / max_impact)
                deviation = (selfish_payoff - Xi) / selfish_payoff

                altruism_data[llm]["relative_harvest_altruism"].append(rel_harvest)
                altruism_data[llm]["marginal_impact_resource"].append(marginal_impact)
                altruism_data[llm]["deviation_from_selfish_nash"].append(deviation)

        # Compute average per LLM
        self.altruism = {
            llm: {measure: sum(values)/len(values) for measure, values in measures.items()}
            for llm, measures in altruism_data.items()
        }

    # -------------------
    # Access methods now return averages
    # -------------------

    def relative_harvest_altruism(self, llm):
        return self.altruism[llm]["relative_harvest_altruism"]

    def marginal_impact_resource(self, llm):
        return self.altruism[llm]["marginal_impact_resource"]

    def deviation_from_selfish_nash(self, llm):
        return self.altruism[llm]["deviation_from_selfish_nash"]

    def get_index(self, llm):
        return self.llm_to_index.get(llm, None)

    def get_llm(self, index):
        return self.index_to_llm.get(index, None)

    def all_indices(self):
        return self.llm_to_index
