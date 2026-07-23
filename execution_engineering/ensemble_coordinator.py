import numpy as np

class EnsembleCoordinator:
    """
    Manages the predictions of multiple models (e.g., the 3 folds or RL ensemble)
    and calibrates their confidence.
    """
    def __init__(self, vote_threshold=2):
        self.vote_threshold = vote_threshold

    def calibrate_confidence(self, prob_wins, pred_oss):
        """
        Determines true confidence by measuring entropy/disagreement among the ensemble.
        """
        mean_prob = np.mean(prob_wins)
        std_prob = np.std(prob_wins)

        # High standard deviation means the models disagree -> Lower confidence
        calibrated_confidence = mean_prob - (std_prob * 0.5)

        # Determine aggregate action
        votes = sum([1 for p in prob_wins if p >= 0.6])

        action = "TRADE" if votes >= self.vote_threshold else "SKIP"

        return action, calibrated_confidence

if __name__ == "__main__":
    print("Ensemble Coordinator defined.")
