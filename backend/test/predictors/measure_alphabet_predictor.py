from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.alphabet_predictor import train_alphabet_predictor

from test.predictors import csv_location

class TestAlphabetPredictor(TestCase):
    def test_accuracy(self) -> None:
        training_data = training_results(csv_location, 2019)
        validation_data = validation_results(csv_location, 2019)
        predictor = train_alphabet_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .35)
