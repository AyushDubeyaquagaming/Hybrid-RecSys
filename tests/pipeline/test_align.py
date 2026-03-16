"""Tests for pipeline/steps/align.py"""
import numpy as np
import pandas as pd
import pytest

from pipeline.steps.align import (
    SCHEMA_COLUMNS,
    align_to_schema,
    is_objectid_like,
    map_device_type,
    map_entry_point,
    map_game_type,
    map_outcome,
    map_provider,
)


class TestMapGameType:
    def test_sic_bo_with_space(self):
        assert map_game_type("sic bo") == "live_dealer"

    def test_sicbo_no_space(self):
        assert map_game_type("sicbo") == "live_dealer"

    def test_aviator_crash(self):
        assert map_game_type("aviator") == "crash"

    def test_baccarat_live_dealer(self):
        assert map_game_type("Baccarat") == "live_dealer"

    def test_roulette_live_dealer(self):
        assert map_game_type("Live Roulette") == "live_dealer"

    def test_slot(self):
        assert map_game_type("starburst slot") == "slot"

    def test_poker_table(self):
        assert map_game_type("poker") == "table"

    def test_unknown(self):
        assert map_game_type("mystery game xyz") == "unknown"

    def test_cricket_crash(self):
        assert map_game_type("cricket crash") == "crash"

    def test_instant_win(self):
        assert map_game_type("instant win keno") == "instant_win"


class TestMapProvider:
    def test_evolution(self):
        assert map_provider("EvoSW") == "Evolution"

    def test_evolution_full(self):
        assert map_provider("evolution") == "Evolution"

    def test_ezugi(self):
        assert map_provider("EZUGI") == "EZUGI"

    def test_spribe(self):
        assert map_provider("spribe") == "Spribe"

    def test_hex_objectid_returns_unknown(self):
        hex_id = "a" * 24
        assert map_provider(hex_id) == "unknown"

    def test_empty_returns_unknown(self):
        assert map_provider("") == "unknown"

    def test_none_string_returns_unknown(self):
        assert map_provider("none") == "unknown"

    def test_hub88(self):
        assert map_provider("hub88") == "HUB88"

    def test_onlyplay(self):
        assert map_provider("onlyplay") == "OnlyPlay"


class TestMapOutcome:
    def test_win_net_positive(self):
        s = pd.Series(["WIN"])
        result = map_outcome(s)
        assert result[0] == "net_positive"

    def test_lose_net_negative(self):
        s = pd.Series(["LOSE"])
        result = map_outcome(s)
        assert result[0] == "net_negative"

    def test_loss_net_negative(self):
        s = pd.Series(["LOSS"])
        result = map_outcome(s)
        assert result[0] == "net_negative"

    def test_other_break_even(self):
        s = pd.Series(["DRAW", "VOID", "REFUND"])
        result = map_outcome(s)
        assert all(r == "break_even" for r in result)

    def test_mixed(self):
        s = pd.Series(["WIN", "LOSS", "LOSE", "DRAW"])
        result = map_outcome(s)
        assert list(result) == [
            "net_positive",
            "net_negative",
            "net_negative",
            "break_even",
        ]


class TestMapDeviceType:
    def test_desktop(self):
        assert map_device_type("desktop") == "desktop"

    def test_android_mobile(self):
        assert map_device_type("Android") == "mobile"

    def test_ios_mobile(self):
        assert map_device_type("iOS") == "mobile"

    def test_tablet(self):
        assert map_device_type("tablet") == "tablet"

    def test_unknown(self):
        assert map_device_type("xyz-device") == "unknown"

    def test_windows_desktop(self):
        assert map_device_type("windows") == "desktop"


class TestAlignToSchemaColumns:
    def _make_transactions(self):
        return pd.DataFrame(
            {
                "userId": ["u1", "u2"],
                "gameName": ["Baccarat", "Roulette"],
                "gameId": ["g1", "g2"],
                "betAmount": [10.0, 20.0],
                "result": ["WIN", "LOSS"],
                "timestamp": pd.to_datetime(["2024-01-01 12:00", "2024-01-02 14:00"]),
                "providerName": ["Evolution", "EZUGI"],
                "categoryName": ["Live Casino", "Live Casino"],
                "session_duration_sec": [300.0, 0.0],
                "entryPoint_raw": ["livecasino", None],
                "deviceType_raw": ["desktop", "Android"],
                "hourOfDay": [12, 14],
                "dayOfWeek": ["Monday", "Tuesday"],
                "providerName_raw": ["Evolution", None],
                "tournamentId": [None, None],
                "win": [1, 0],
                "minBet": [1.0, 1.0],
                "maxBet": [100.0, 100.0],
            }
        )

    def test_output_has_all_schema_columns(self):
        transactions_df = self._make_transactions()
        result = align_to_schema.fn(transactions_df)
        for col in SCHEMA_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_has_exactly_schema_columns(self):
        transactions_df = self._make_transactions()
        result = align_to_schema.fn(transactions_df)
        assert list(result.columns) == SCHEMA_COLUMNS

    def test_no_null_user_ids(self):
        transactions_df = self._make_transactions()
        result = align_to_schema.fn(transactions_df)
        assert result["userId"].notna().all()

    def test_event_type_constant(self):
        transactions_df = self._make_transactions()
        result = align_to_schema.fn(transactions_df)
        assert (result["eventType"] == "game_session").all()
