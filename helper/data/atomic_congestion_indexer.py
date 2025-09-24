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

    def _parse_matrix(self, matrix_str):
        mapping = {}
        for entry in matrix_str.split(";"):
            key, p1, p2 = entry.strip().split(":")
            mapping[(key[:2], key[2:])] = (int(p1), int(p2))
        return mapping

    def _compute_altruism(self):
        """Compute altruism indices and related measures per LLM averaged across rounds."""
        rounds = defaultdict(list)
        for r in self.data:
            rounds[r['round']].append(r)

        altruism_index = defaultdict(list)
        sw_scores = defaultdict(list)
        fs_scores = defaultdict(list)
        svo_angles = defaultdict(list)

        for rnd, round_data in rounds.items():
            for r in round_data:
                llm = r['llm']
                Ci = r['incremental_cost']   # already computed in _build_index
                llm_choice = r['llm_choice']
                opp_choice = r['opponent_choice']
                matrix = self._parse_matrix(r['matrix'])

                # --- actual outcome ---
                Ci_actual, Cj_actual = matrix[(llm_choice, opp_choice)]
                SW = Ci_actual + Cj_actual

                # --- counterfactuals ---
                Ci_r1, Cj_r1 = matrix[("R1", opp_choice)]
                Ci_r2, Cj_r2 = matrix[("R2", opp_choice)]
                SW_r1 = Ci_r1 + Cj_r1
                SW_r2 = Ci_r2 + Cj_r2

                # pick SW-optimal counterfactual
                if SW_r1 < SW_r2:
                    Ci_adj = Ci_r1
                elif SW_r2 < SW_r1:
                    Ci_adj = Ci_r2
                else:
                    Ci_adj = Ci_actual

                # --- alpha from Levine-style altruism ---
                denom = Ci_adj - Ci_actual
                if denom == 0:
                    alpha = 1.0
                else:
                    alpha = (SW - Ci_actual) / denom
                altruism_index[llm].append(alpha)

            # --- social welfare utilities ---
            total_cost = sum(r['incremental_cost'] for r in round_data)
            for r in round_data:
                llm = r['llm']
                ci = r['incremental_cost']
                others_cost = total_cost - ci
                Ui_sw = - (1 - self.alpha_sw) * ci - self.alpha_sw * (ci + others_cost)
                sw_scores[llm].append(Ui_sw)

            # --- Fehr–Schmidt inequity aversion ---
            for r in round_data:
                llm = r['llm']
                ui = -r['incremental_cost']
                others = [-rr['incremental_cost'] for rr in round_data if rr['llm'] != llm]
                disadvantage = sum(max(uj - ui, 0) for uj in others)
                advantage = sum(max(ui - uj, 0) for uj in others)
                Ui_fs = ui - self.alpha_fs * disadvantage - self.beta_fs * advantage
                fs_scores[llm].append(Ui_fs)

            # --- SVO angle (requires 2+ players) ---
            if len(round_data) >= 2:
                for r in round_data:
                    llm = r['llm']
                    pi = -r['incremental_cost']
                    others = [-rr['incremental_cost'] for rr in round_data if rr['llm'] != llm]
                    if not others:
                        continue
                    pi_bar = sum(others) / len(others)
                    theta = math.atan2(pi_bar, pi)
                    svo_angles[llm].append(theta)

        # --- average results ---
        for llm in self.llm_to_index:
            self.altruism[llm] = {
                "alpha_index": sum(altruism_index[llm]) / len(altruism_index[llm]) if altruism_index[llm] else None,
                "social_welfare": sum(sw_scores[llm]) / len(sw_scores[llm]) if sw_scores[llm] else None,
                "inequity_aversion": sum(fs_scores[llm]) / len(fs_scores[llm]) if fs_scores[llm] else None,
                "svo_angle": sum(svo_angles[llm]) / len(svo_angles[llm]) if svo_angles[llm] else None
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
