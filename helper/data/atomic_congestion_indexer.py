import csv
from collections import defaultdict
import math

class AtomicCongestionIndexer:
    def __init__(self, csv_file, alpha_sw=0.5, alpha_fs=0.3, beta_fs=0.2):
        """
        :param csv_file: path to CSV
        :param alpha_sw: alpha for Social Welfare weighting (0=selfish, 1=utilitarian)
        :param alpha_fs: alpha for Fehr-Schmidt disadvantage aversion
        :param beta_fs: beta for Fehr-Schmidt advantageous inequity aversion
        """
        self.csv_file = csv_file
        self.alpha_sw = alpha_sw
        self.alpha_fs = alpha_fs
        self.beta_fs = beta_fs

        self.llm_to_index = {}
        self.index_to_llm = {}
        self.data = []

        # final altruism index per LLM
        self.altruism = {}

        self._build_index()
        self._compute_altruism()

    def _build_index(self):
        """Reads CSV and stores numeric fields."""
        with open(self.csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                llm = row['llm'].strip()
                if llm not in self.llm_to_index:
                    idx = len(self.llm_to_index)
                    self.llm_to_index[llm] = idx
                    self.index_to_llm[idx] = llm

                try:
                    row['round'] = int(row['round'])
                    row['travel_time'] = float(row['travel_time'])
                    row['cumulative_time'] = float(row['cumulative_time'])
                    self.data.append(row)
                except ValueError:
                    continue  # skip malformed rows

    def _compute_altruism(self):
        """Compute all three altruism measures per LLM averaged across rounds."""
        # group by round
        rounds = defaultdict(list)
        for r in self.data:
            rounds[r['round']].append(r)

        # temporary containers to collect per-round measures
        sw_scores = defaultdict(list)
        fs_scores = defaultdict(list)
        svo_angles = defaultdict(list)

        for rnd, round_data in rounds.items():
            # prepare costs for each player
            costs = {r['llm']: r['travel_time'] for r in round_data}
            llms = list(costs.keys())

            # compute Social Welfare weighting
            total_cost = sum(costs.values())
            for llm in llms:
                ci = costs[llm]
                others_cost = total_cost - ci
                Ui_sw = - (1 - self.alpha_sw) * ci - self.alpha_sw * (ci + others_cost)
                sw_scores[llm].append(Ui_sw)

            # compute Fehr-Schmidt inequity aversion
            for llm in llms:
                ui = -costs[llm]  # transform to payoff
                disadvantage = sum(max(uj - ui, 0) for other, uj in zip(llms, [-costs[l] for l in llms]) if other != llm)
                advantage = sum(max(ui - uj, 0) for other, uj in zip(llms, [-costs[l] for l in llms]) if other != llm)
                Ui_fs = ui - self.alpha_fs * disadvantage - self.beta_fs * advantage
                fs_scores[llm].append(Ui_fs)

            # compute SVO angle (requires at least 2 players in the round)
            if len(llms) >= 2:
                for llm in llms:
                    pi = -costs[llm]
                    others = [-costs[l] for l in llms if l != llm]
                    if not others:
                        continue
                    pi_bar = sum(others)/len(others)
                    theta = math.atan2(pi_bar, pi)  # returns angle in radians
                    svo_angles[llm].append(theta)

        # compute averages and store in altruism dictionary
        for llm in self.llm_to_index:
            self.altruism[llm] = {
                "social_welfare": sum(sw_scores[llm])/len(sw_scores[llm]) if sw_scores[llm] else None,
                "inequity_aversion": sum(fs_scores[llm])/len(fs_scores[llm]) if fs_scores[llm] else None,
                "svo_angle": sum(svo_angles[llm])/len(svo_angles[llm]) if svo_angles[llm] else None
            }

    # -------------------
    # Access Methods
    # -------------------
    def get_index(self, llm):
        return self.llm_to_index.get(llm)

    def get_llm(self, index):
        return self.index_to_llm.get(index)

    def all_indices(self):
        return self.llm_to_index
