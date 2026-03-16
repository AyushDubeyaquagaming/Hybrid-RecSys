"""Tests for pipeline/steps/ingest.py"""
import pandas as pd
import pytest

from pipeline.steps.ingest import clean_and_merge, extract_bet_part


class TestExtractBetPart:
    def test_extracts_sport_category_tournament(self):
        bet_parts = [
            {
                "sportName": "Baccarat",
                "categoryName": "Live Casino",
                "tournamentName": "Evolution",
                "tournamentId": "t123",
            }
        ]
        result = extract_bet_part(bet_parts)
        assert result["sportName"] == "Baccarat"
        assert result["categoryName"] == "Live Casino"
        assert result["tournamentName"] == "Evolution"
        assert result["tournamentId"] == "t123"

    def test_empty_bet_parts_returns_nones(self):
        result = extract_bet_part([])
        assert result["sportName"] is None
        assert result["categoryName"] is None
        assert result["tournamentName"] is None
        assert result["tournamentId"] is None

    def test_none_bet_parts_returns_nones(self):
        result = extract_bet_part(None)
        assert result["sportName"] is None

    def test_partial_fields_returns_none_for_missing(self):
        result = extract_bet_part([{"sportName": "Roulette"}])
        assert result["sportName"] == "Roulette"
        assert result["categoryName"] is None


class TestGameNameNormalization:
    def _make_transactions(self, game_names):
        return pd.DataFrame(
            {
                "userId": ["u1"] * len(game_names),
                "gameName": game_names,
                "betAmount": [10.0] * len(game_names),
                "result": ["WIN"] * len(game_names),
                "timestamp": pd.Timestamp("2024-01-01"),
                "providerName_raw": [None] * len(game_names),
                "categoryName": ["Live Casino"] * len(game_names),
                "tournamentId": [None] * len(game_names),
                "hourOfDay": [12] * len(game_names),
                "dayOfWeek": ["Monday"] * len(game_names),
            }
        )

    def _make_games(self, game_name):
        return pd.DataFrame(
            {
                "gameName": [game_name],
                "gameId": ["g1"],
                "gamevendor": ["Evolution"],
                "minBet": [1.0],
                "maxBet": [100.0],
            }
        )

    def test_baccarat_normalization(self):
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        transactions_df = self._make_transactions(["998:baccarat"])
        games_df = self._make_games("Baccarat")
        result = clean_and_merge.fn(transactions_df, games_df, settings)
        assert result["gameName"].iloc[0] == "Baccarat"

    def test_football_studio_normalization(self):
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        transactions_df = self._make_transactions(["Football studio"])
        games_df = self._make_games("Football Studio")
        result = clean_and_merge.fn(transactions_df, games_df, settings)
        assert result["gameName"].iloc[0] == "Football Studio"


class TestCleanAndMergeNoRowInflation:
    def test_no_row_inflation(self):
        """Merging games_df must not inflate transaction rows."""
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        transactions_df = pd.DataFrame(
            {
                "userId": ["u1", "u2", "u3"],
                "gameName": ["Baccarat", "Baccarat", "Roulette"],
                "betAmount": [10.0, 20.0, 15.0],
                "result": ["WIN", "LOSS", "WIN"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "providerName_raw": ["Evolution", None, "EZUGI"],
                "categoryName": ["Live Casino"] * 3,
                "tournamentId": [None] * 3,
                "hourOfDay": [12, 14, 16],
                "dayOfWeek": ["Monday", "Tuesday", "Wednesday"],
            }
        )
        # games_df intentionally has the same gameName twice (de-dup must be applied)
        games_df = pd.DataFrame(
            {
                "gameName": ["Baccarat", "Baccarat", "Roulette"],
                "gameId": ["g1", "g1", "g2"],
                "gamevendor": ["Evolution", "Evolution", "EZUGI"],
                "minBet": [1.0, 1.0, 2.0],
                "maxBet": [100.0, 100.0, 200.0],
            }
        )
        result = clean_and_merge.fn(transactions_df, games_df, settings)
        assert len(result) == len(transactions_df), (
            f"Row inflation detected: {len(transactions_df)} -> {len(result)}"
        )
