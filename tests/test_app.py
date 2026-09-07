from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def button(app, label):
    return next(b for b in app.button if b.label == label)


def demo():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    app.checkbox(key="demo").check().run(timeout=30)
    assert not app.exception
    return app


def test_all_demo_views_render_without_live_requests(monkeypatch):
    from utils import services

    def forbidden(*args, **kwargs):
        raise AssertionError("Demo should never request live data")

    for name in ["market", "news", "compliance"]:
        monkeypatch.setattr(services, name, forbidden)
    app = demo()
    for page in ["Charts", "Evaluate", "Screener", "News", "Overview"]:
        app.radio(key="page").set_value(page).run(timeout=30)
        assert not app.exception, page


def test_ticker_change_cannot_show_or_export_previous_backtest():
    app = demo()
    app.radio(key="page").set_value("Evaluate").run()
    button(app, "Run historical evaluation").click().run(timeout=30)
    assert not app.exception and "evaluation" in app.session_state
    app.text_input(key="ticker").set_value("MSFT").run(timeout=30)
    assert not app.exception and "evaluation" not in app.session_state
    assert any("Run an evaluation for this ticker" in i.value for i in app.info)
    assert not any(e.proto.label == "Download holdout trades" for e in app.get("download_button"))


def test_settings_change_hides_results_and_refresh_clears_them():
    app = demo()
    app.radio(key="page").set_value("Evaluate").run()
    button(app, "Run historical evaluation").click().run(timeout=30)
    app.number_input(key="e_commission").set_value(1.0).run()
    assert not app.exception and any(
        "Run an evaluation for this ticker" in i.value for i in app.info
    )
    button(app, "Load / refresh data").click().run()
    assert not app.exception and "evaluation" not in app.session_state


def test_invalid_rules_show_error_not_traceback():
    app = demo()
    app.number_input(key="s_rsi_min").set_value(99.0).run()
    assert not app.exception and any("RSI minimum" in e.value for e in app.error)


def test_failed_provider_clears_previous_stock(monkeypatch):
    from utils import services

    app = demo()

    def failure(*args, **kwargs):
        raise ValueError("Bad bars")

    monkeypatch.setattr(services, "market", failure)
    app.checkbox(key="demo").uncheck().run()
    button(app, "Load / refresh data").click().run()
    assert not app.exception and "market" not in app.session_state
    assert any("could not be loaded" in e.value for e in app.error)


def test_lab_demo_freeze_and_reveal():
    app = demo()
    app.radio(key="page").set_value("Strategy Lab").run(timeout=30)
    button(app, "Prepare and freeze study").click().run(timeout=60)
    assert not app.exception
    assert "lab_study" in app.session_state
    assert "lab_result" not in app.session_state
    button(app, "Reveal final test once").click().run(timeout=60)
    assert not app.exception
    assert "lab_result" in app.session_state
    app.number_input(key="lab_fee").set_value(2.0).run()
    assert "lab_result" not in app.session_state
