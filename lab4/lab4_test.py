import math

import pytest

import lab4

# запуск из папки lab4: python -m pytest -v lab4_test.py

def test_only_available_asset_can_be_sold_when_other_assets_at_minimum():
    """ если два актива находятся на минимуме, продавать их нельзя """
    state = (100.0, 150.0, 100.0, 600.0)

    actions = lab4.make_actions(state)

    assert actions
    assert all(action[1] >= 0 for _, action in actions)
    assert all(action[2] >= 0 for _, action in actions)
    assert any(action[0] < 0 for _, action in actions)


def test_zero_assets_and_zero_cash_still_allow_idle_action():
    """ при нулевых доступных ресурсах бездействие остается допустимым """
    state = (0.0, 0.0, 0.0, 0.0)

    actions = lab4.make_actions(state)

    assert actions == [((0.0, 0.0, 0.0, 0.0), (0, 0, 0))]


def test_multiple_optimal_actions_are_preserved(monkeypatch):
    """ при равных результатах функция последнего этапа возвращает все оптимумы """
    monkeypatch.setattr(lab4, "FEES", (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        lab4,
        "SCENARIOS",
        tuple((((1.0, (1.0, 1.0, 1.0))),) for _ in range(3)),
    )

    state = (100.0, 800.0, 400.0, 600.0)
    value, actions = lab4.terminal_stage_value(state)

    assert math.isfinite(value)
    assert len(actions) > 1
    assert any(action == (0, 0, 0) for action, _ in actions)


def test_zero_returns_make_no_trade_optimal(monkeypatch):
    """ при нулевой доходности активов без комиссии оптимально сохранить портфель """
    monkeypatch.setattr(lab4, "FEES", (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        lab4,
        "SCENARIOS",
        tuple((((1.0, (1.0, 1.0, 1.0))),) for _ in range(3)),
    )

    state = (100.0, 800.0, 400.0, 600.0)
    value, actions = lab4.terminal_stage_value(state)

    assert value == pytest.approx(sum(state))
    assert any(action == (0, 0, 0) for action, _ in actions)


def test_validator_accepts_initial_parameters():
    """ исходные параметры задачи должны проходить проверку """
    lab4.validate_input()


def test_validator_rejects_negative_initial_cash(monkeypatch):
    """ отрицательная начальная сумма денег должна приводить к ошибке """
    monkeypatch.setattr(lab4, "INITIAL_CASH", -1.0)

    with pytest.raises(ValueError):
        lab4.validate_input()


def test_validator_rejects_zero_lot(monkeypatch):
    """ размер пакета не может быть нулевым"""
    monkeypatch.setattr(lab4, "LOTS", (25.0, 0.0, 100.0))

    with pytest.raises(ValueError):
        lab4.validate_input()


def test_validator_rejects_invalid_fee(monkeypatch):
    """ комиссия должна быть неотрицательной и меньше единицы """
    monkeypatch.setattr(lab4, "FEES", (0.04, 1.0, 0.05))

    with pytest.raises(ValueError):
        lab4.validate_input()


def test_validator_rejects_probabilities_that_do_not_sum_to_one(monkeypatch):
    """ сумма вероятностей сценариев каждого этапа должна равняться единице """
    scenarios = list(lab4.SCENARIOS)
    scenarios[0] = (
        (0.5, (1.20, 1.10, 1.07)),
        (0.3, (1.05, 1.02, 1.03)),
        (0.1, (0.80, 0.95, 1.00)),
    )
    monkeypatch.setattr(lab4, "SCENARIOS", tuple(scenarios))

    with pytest.raises(ValueError):
        lab4.validate_input()


def test_bellman_returns_finite_value_and_optimal_action():
    """ функция Беллмана должна возвращать конечное значение и оптимальное управление """
    state = (*lab4.INITIAL_ASSETS, lab4.INITIAL_CASH)

    value, actions = lab4.bellman(0, state)

    assert math.isfinite(value)
    assert actions
    assert all(len(action) == 2 for action in actions)


def test_initial_optimum_matches_manual_calculation():
    """ проверяем начальное оптимальное значение по ручному расчету из отчета """
    state = (*lab4.INITIAL_ASSETS, lab4.INITIAL_CASH)

    value, actions = lab4.bellman(0, state)

    assert value == pytest.approx(2034.238884, abs=1e-5)
    assert actions[0][0] == (23, 0, 0)


def test_pruning_matches_full_enumeration_for_sample_state(monkeypatch):
    """ отсечение не должно менять оптимум для контрольного состояния"""
    state = (810.0, 880.0, 428.0, 2.0)

    monkeypatch.setattr(lab4, "USE_PRUNING", True)
    pruned_value, pruned_actions = lab4.bellman(1, state)

    lab4.bellman.cache_clear()
    monkeypatch.setattr(lab4, "USE_PRUNING", False)
    full_value, full_actions = lab4.bellman(1, state)

    assert pruned_value == pytest.approx(full_value, abs=1e-8)
    assert pruned_actions == full_actions