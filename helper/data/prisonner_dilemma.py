import csv
from collections import defaultdict

class PrisonersDilemmaIndexer:
    def __init__(self, csv_file, T=5, R=3, P=1, S=0):
        """
        :param csv_file: path to CSV containing PD play data
        :param T: Temptation payoff (defect vs cooperator)
        :param R: Reward for mutual cooperation
        :param P: Punishment for mutual defection
        :param S: Sucker's payoff (cooperate vs defector)

        Expected CSV format:
        round,llm,llm_choice,opponent_choice,reasoning,travel_time,cumulative_time
        """
        self.csv_file = csv_file
        self.T, self.R, self.P, self.S = T, R, P, S

        self.llm_to_index = {}
        self.index_to_llm = {}
        self.data = []

        # final altruism index per LLM
        self.altruism = {}

        self._build_index()
        self._compute_altruism()

    def _build_index(self):
        """Reads CSV and stores PD-relevant fields."""
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
                    row['llm_choice'] = row['llm_move'].strip().upper()
                    row['opponent_choice'] = row['opponent_move'].strip().upper()
                    if row['llm_choice'] not in ['C', 'D'] or row['opponent_choice'] not in ['C', 'D']:
                        continue
                    self.data.append(row)
                except ValueError:
                    continue  # skip malformed rows

    def _compute_altruism(self):
        """
        Compute three PD-based altruism measures per LLM:
        1. Cooperation frequency above Nash (baseline 0 for one-shot PD)
        2. Payoff-sacrifice ratio (T-R for cooperating vs defecting)
        3. Mutual cooperation sustainability in repeated rounds
        """
        rounds = defaultdict(list)
        for r in self.data:
            rounds[r['round']].append(r)

        # temporary containers to collect per-round measures
        cooperation_counts = defaultdict(int)
        total_rounds = defaultdict(int)
        payoff_sacrifices = defaultdict(list)
        mutual_coop_counts = defaultdict(int)
        coop_following_coop_counts = defaultdict(int)

        # sort rounds in ascending order
        sorted_rounds = sorted(rounds.items())

        # track previous choice of opponents per LLM for mutual cooperation sustainability
        previous_choice = defaultdict(lambda: None)

        for rnd, round_data in sorted_rounds:
            for r in round_data:
                llm = r['llm']
                choice = r['llm_choice']
                opp_choice = r['opponent_choice']

                total_rounds[llm] += 1
                if choice == 'C':
                    cooperation_counts[llm] += 1
                    # payoff sacrifice relative to defecting against cooperator
                    if opp_choice == 'C':
                        payoff_sacrifices[llm].append(self.T - self.R)
                    elif opp_choice == 'D':
                        payoff_sacrifices[llm].append(self.S - self.P)  # negative or zero

                # mutual cooperation tracking
                if choice == 'C' and opp_choice == 'C':
                    mutual_coop_counts[llm] += 1
                    # check if this follows previous round where opponent cooperated
                    if previous_choice[llm] == 'C':
                        coop_following_coop_counts[llm] += 1

                previous_choice[llm] = choice

        for llm in self.llm_to_index:
            coop_freq = cooperation_counts[llm] / total_rounds[llm] if total_rounds[llm] > 0 else None
            avg_sacrifice = sum(payoff_sacrifices[llm])/len(payoff_sacrifices[llm]) if payoff_sacrifices[llm] else None
            coop_sustain = coop_following_coop_counts[llm]/mutual_coop_counts[llm] if mutual_coop_counts[llm] > 0 else None

            self.altruism[llm] = {
                "cooperation_frequency": coop_freq,
                "avg_payoff_sacrifice": avg_sacrifice,
                "mutual_cooperation_sustainability": coop_sustain
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

    def get_altruism(self, llm):
        return self.altruism.get(llm)
