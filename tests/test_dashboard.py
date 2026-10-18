from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_controls_and_missing_history():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == 'Battery fleet health'
    assert len(at.dataframe) >= 4
    today = next(x for x in at.slider if x.label == 'Today cycle')
    today.set_value(0).run(timeout=30)
    assert not at.exception
    assert next(x for x in at.metric if x.label == 'No current data').value == '7'
    at.multiselect[0].set_value(['oxford']).run(timeout=30)
    assert not at.exception
    at.radio[0].set_value('Entire dataset held out').run(timeout=30)
    assert not at.exception
    at.radio[1].set_value('Trailing median').run(timeout=30)
    assert not at.exception
    next(x for x in at.selectbox if x.label == 'RUL trend model').set_value('exponential').run(timeout=30)
    assert not at.exception


def test_improvement_views_only_use_saved_held_out_artifacts():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run(timeout=30)
    control = next(x for x in at.selectbox if x.label == 'Model view')
    control.set_value('Inner-selected features (E8; known history)').run(timeout=30)
    assert not at.exception
    next(x for x in at.selectbox if x.label == 'Model view').set_value('Probabilistic tracker (E5/E6; experimental)').run(timeout=30)
    assert not at.exception
    next(x for x in at.selectbox if x.label == 'Probability alert horizon (lab cycles)').set_value(180).run(timeout=30)
    assert not at.exception
    assert any('not operationally calibrated' in x.value for x in at.warning)
    next(x for x in at.slider if x.label == 'Today cycle').set_value(0).run(timeout=30)
    assert not at.exception
    assert next(x for x in at.metric if x.label == 'No current data').value == '7'
