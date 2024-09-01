from typing import Iterable
from matchpredictor.matchresults.result import Fixture, Outcome, Result
from matchpredictor.predictors.predictor import Prediction, Predictor


class HomeWinPredictor(Predictor):
    def __init__(self, results: Iterable[Result]) -> None:
        self.results = results

    def predict(self, fixture: Fixture) -> Prediction:
        home_win_prob = self._calculate_home_win_probability(fixture)

        if home_win_prob > 0.5:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.AWAY if home_win_prob < 0.5 else Outcome.DRAW)

    def _calculate_home_win_probability(self, fixture: Fixture) -> float:
        # Filter the historical results for matches involving the home team
        home_team_results = [result for result in self.results if result.fixture.home_team == fixture.home_team]
        home_wins = sum(1 for result in home_team_results if result.home_goals > result.away_goals)
        total_games = len(home_team_results)

        # Return the win probability, default to 0.5 if no historical data is available
        return home_wins / total_games if total_games > 0 else 0.5


def train_home_win_predictor(results: Iterable[Result]) -> Predictor:
    return HomeWinPredictor(results)
