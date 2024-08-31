import random
from typing import Iterable

from matchpredictor.matchresults.result import Fixture, Outcome, Result
from matchpredictor.predictors.predictor import Predictor, Prediction



class AlphabetPredictor(Predictor):
    def __init__(self, results: Iterable[Result]) -> None:
        self.results = results

    def predict(self, fixture: Fixture) -> Prediction:
        winning_team = min([fixture.home_team, fixture.away_team], key=lambda team: team.name)

        if winning_team.name == fixture.home_team.name:
            return Prediction(outcome=Outcome.HOME)
        elif winning_team.name == fixture.away_team.name:
            return Prediction(outcome=Outcome.AWAY)

        return Prediction(outcome=Outcome.DRAW)


def train_alphabet_predictor(results: Iterable[Result]) -> Predictor:
    return AlphabetPredictor(results)
