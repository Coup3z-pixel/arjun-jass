import csv
from collections import defaultdict
from datetime import datetime

class CostSharingSchedulerIndexer:
    def __init__(self, csv_file):
        self.csv_file = csv_file

        self.llm_to_index = {}
        self.index_to_llm = {}
        self.data = []

        self.altruism = {}
        self.utility = {}

        self._build_index()
        self._compute_measures()

    # -------------------
    # CSV Reading & Parsing
    # -------------------
    def _build_index(self):
        with open(self.csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                llm = row['llm_name'].strip()
                if llm not in self.llm_to_index:
                    idx = len(self.llm_to_index)
                    self.llm_to_index[llm] = idx
                    self.index_to_llm[idx] = llm

                try:
                    row['individual_time'] = self.parse_time(row['individual_time'])
                    row['team_time'] = self.parse_time(row['team_time'])
                    row['individual_payout'] = float(row['individual_payout'])
                    row['team_payout'] = float(row['team_payout'])
                    self.data.append(row)
                except ValueError as e:
                    print(f"Skipping row due to conversion error: {row}, Error: {e}")
                    continue

    def parse_time(self, t):
        if isinstance(t, str) and t.strip():
            dt = datetime.strptime(t.strip(), "%I:%M %p")
            return dt.hour * 60 + dt.minute
        return None

    # -------------------
    # Computation
    # -------------------
    def _compute_measures(self):
        altruism_eq13 = defaultdict(list)
        altruism_eq14 = defaultdict(list)
        utilities = defaultdict(list)

        # Precompute per-LLM min_Ci (individual_time)
        min_ci_per_llm = {}
        for r in self.data:
            llm = r['llm_name']
            Ci = r['individual_time']
            if Ci is not None:
                if llm not in min_ci_per_llm:
                    min_ci_per_llm[llm] = Ci
                else:
                    min_ci_per_llm[llm] = min(min_ci_per_llm[llm], Ci)

        # Iterate once over data
        for r in self.data:
            llm = r['llm_name']
            Ci = r['individual_time']
            Ti = r['team_time']
            Ei = r['individual_payout']
            Ti_payout = r['team_payout']

            # eq13: altruism wrt min individual time
            if Ei is not None and Ci is not None and Ei != min_ci_per_llm[llm]:
                Ai_13 = (Ei - Ci) / (Ei - min_ci_per_llm[llm])
                altruism_eq13[llm].append(Ai_13)

            # eq14: altruism wrt team vs individual time
            if Ti is not None and Ci is not None and Ti > 0:
                Ai_14 = (Ti - Ci) / Ti
                altruism_eq14[llm].append(Ai_14)

            # simple utility: convex combo of Ei and team payout
            alpha = 0.5
            if Ei is not None and Ti_payout is not None:
                U = (1 - alpha) * Ei + alpha * Ti_payout
                utilities[llm].append(U)

        # Aggregate results per LLM
        for llm in self.llm_to_index:
            self.altruism[llm] = {
                "eq13": sum(altruism_eq13[llm]) / len(altruism_eq13[llm]) if altruism_eq13[llm] else None,
                "eq14": sum(altruism_eq14[llm]) / len(altruism_eq14[llm]) if altruism_eq14[llm] else None
            }
            self.utility[llm] = sum(utilities[llm]) / len(utilities[llm]) if utilities[llm] else None

    # -------------------
    # Access Methods
    # -------------------
    def get_index(self, llm):
        return self.llm_to_index.get(llm)

    def get_llm(self, index):
        return self.index_to_llm.get(index)

    def all_indices(self):
        return list(self.llm_to_index.keys())

    def get_altruism(self, llm):
        return self.altruism.get(llm)

    def get_utility(self, llm):
        return self.utility.get(llm)
