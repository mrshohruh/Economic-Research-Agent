import copy
import json
from unittest.mock import Mock
import pandas as pd
import pytest
from uzhousing.llm import LLMUnavailable
from uzhousing.report.olx_bulletin import Section, Text, Bullets, Tbl, Note, TITLES
from uzhousing.report.olx_narrative import enrich


def content():
    return [Section(TITLES['summary'], [Bullets(['Median 500 USD.'])]),
            Section(TITLES['rent'], [Text('Oylik ijara.'), Tbl('USD/oy', pd.DataFrame({'Hudud':['Toshkent'], 'Mediana':[500]})), Note('Eslatma',['Bu bitim narxi emas.'])])]


def fake():
    llm = Mock(available=True, model='test-model', last_usage={'input_tokens':100,'output_tokens':50})
    def response(prompt, **kwargs):
        data = json.loads(prompt)
        fact = next(k for k,v in data['fact_values'].items() if v=='500')
        return {'blocks':[{'id':t['id'], 'items':['Oylik ijara taklifi '+fact+' USD.']*t['item_count']} for t in data['targets']]}
    llm.complete_json.side_effect = response
    return llm


def test_claude_changes_prose_only():
    sections = content()
    original = copy.deepcopy(sections)
    meta = enrich(sections, fake())
    assert meta['generated_by'] == 'anthropic_claude'
    assert meta['usage']['output_tokens'] == 50
    assert sections[0].blocks[0].items == ['Oylik ijara taklifi 500 USD.']
    assert [s.title for s in sections] == [s.title for s in original]
    assert [[type(b) for b in s.blocks] for s in sections] == [[type(b) for b in s.blocks] for s in original]
    pd.testing.assert_frame_equal(sections[1].blocks[1].frame, original[1].blocks[1].frame)
    assert sections[1].blocks[2] == original[1].blocks[2]


@pytest.mark.parametrize('bad', ['Narx 999 USD.', 'Narx [[F999999]] USD.', '', None])
def test_invalid_prose_does_not_partially_mutate(bad):
    sections=content()
    llm=fake()
    llm.complete_json.side_effect=None
    llm.complete_json.return_value={'blocks':[{'id':'s0b0','items':['Yangi xulosa.']},{'id':'s1b0','items':[bad]}]}
    with pytest.raises(LLMUnavailable):
        enrich(sections,llm)
    assert sections[0].blocks[0].items == ['Median 500 USD.']


def test_missing_key_and_service_error_fail():
    llm=fake()
    llm.available=False
    with pytest.raises(LLMUnavailable):
        enrich(content(),llm)
    llm.complete_json.assert_not_called()
    llm.available=True
    llm.complete_json.side_effect=LLMUnavailable('service error')
    with pytest.raises(LLMUnavailable):
        enrich(content(),llm)


def test_missing_or_reordered_blocks_rejected():
    llm=fake()
    llm.complete_json.side_effect=None
    llm.complete_json.return_value={'blocks':[]}
    with pytest.raises(LLMUnavailable):
        enrich(content(),llm)


def test_missing_key_stops_before_collection(tmp_path, monkeypatch):
    from uzhousing.config import Settings
    from uzhousing.report.olx_bulletin import run_olx
    from uzhousing.ingest import olx
    collect=Mock()
    monkeypatch.setattr(olx, 'collect', collect)
    with pytest.raises(LLMUnavailable):
        run_olx(Settings(anthropic_api_key='', output_dir=tmp_path), progress=lambda _:None)
    collect.assert_not_called()


def test_method_authorship_updated_without_new_blocks():
    sections=content()+[Section(TITLES['method'],[Bullets(["Hisob-kitoblar dastur orqali bajarildi; matn o'zbek tilidagi shablon asosida yozildi."])])]
    enrich(sections,fake())
    assert len(sections[-1].blocks)==1
    assert len(sections[-1].blocks[0].items)==1
    assert 'Anthropic Claude' in sections[-1].blocks[0].items[0]


def evidence_of(llm):
    """The prompt the model was sent, decoded."""
    return json.loads(llm.complete_json.call_args[0][0])


def test_prose_decimals_and_periods_are_masked_as_single_facts():
    sections = [Section(TITLES['summary'],
                        [Bullets(['Mediana 20,37 mln so\'m/m², 2026-yil II choragida.'])])]
    llm = fake()
    captured = {}

    def response(prompt, **kwargs):
        captured.update(json.loads(prompt))
        return {'blocks': [{'id': t['id'], 'items': ['Matn.'] * t['item_count']}
                           for t in captured['targets']]}
    llm.complete_json.side_effect = response
    enrich(sections, llm)
    values = set(captured['fact_values'].values())
    # A comma decimal is one number, and the whole period phrase is one fact,
    # so the model cannot pair a year with a quarter of its own choosing.
    assert '20,37' in values and '2026-yil II choragida' in values


def test_table_cells_are_masked_one_value_at_a_time():
    frame = pd.DataFrame({'Hudud': ['Toshkent'], "E'lonlar": [1025], 'Mediana': [17.51]})
    sections = [Section(TITLES['rent'], [Bullets(['Matn.']), Tbl('Jadval', frame)])]
    llm = fake()
    captured = {}

    def response(prompt, **kwargs):
        captured.update(json.loads(prompt))
        return {'blocks': [{'id': t['id'], 'items': ['Matn.'] * t['item_count']}
                           for t in captured['targets']]}
    llm.complete_json.side_effect = response
    enrich(sections, llm)
    values = set(captured['fact_values'].values())
    assert {'1025', '17.51'} <= values and '1025,17.51' not in values


def test_a_quarter_the_model_invented_is_refused():
    sections = content()
    llm = fake()
    llm.complete_json.side_effect = None
    llm.complete_json.return_value = {'blocks': [
        {'id': 's0b0', 'items': ['III chorakda narx oshdi.']},
        {'id': 's1b0', 'items': ['Matn.']}]}
    with pytest.raises(LLMUnavailable):
        enrich(sections, llm)
    assert sections[0].blocks[0].items == ['Median 500 USD.']


def test_flagged_paragraphs_are_sent_back_for_one_revision_round():
    sections = content()
    llm = fake()
    calls = []

    def response(prompt, **kwargs):
        data = json.loads(prompt)
        calls.append(data)
        worn = 'Narx darajalar bo\'yicha o\'zgardi.'
        text = worn if len(calls) == 1 else 'Qayta yozilgan xatboshi, chunki ma\'nosi bor.'
        return {'blocks': [{'id': t['id'], 'items': [text] * t['item_count']}
                           for t in data['targets']]}
    llm.complete_json.side_effect = response
    meta = enrich(sections, llm)
    assert len(calls) == 2 and 'revise' in calls[1]
    assert meta['editorial']['revised_blocks'] == len(calls[1]['targets'])
    assert sections[0].blocks[0].items == ['Qayta yozilgan xatboshi, chunki ma\'nosi bor.']
    assert meta['editorial']['remaining']['by_code'].get('phrase') is None


def test_a_failed_revision_keeps_the_accepted_first_draft():
    sections = content()
    llm = fake()
    state = {'calls': 0}

    def response(prompt, **kwargs):
        state['calls'] += 1
        if state['calls'] > 1:
            raise LLMUnavailable('service error')
        data = json.loads(prompt)
        return {'blocks': [{'id': t['id'], 'items': ["Narx darajalar bo'yicha o'zgardi."]
                            * t['item_count']} for t in data['targets']]}
    llm.complete_json.side_effect = response
    meta = enrich(sections, llm)
    assert meta['editorial']['revised_blocks'] == 0
    assert sections[0].blocks[0].items == ["Narx darajalar bo'yicha o'zgardi."]


def test_a_period_and_the_numbers_beside_it_are_masked_in_one_pass():
    # Two passes nested a reference inside another ([[F[[F46]]]]), and the
    # model copying the inner one printed a literal [[F46]] in the report.
    from uzhousing.report.olx_narrative import PROSE_MASK
    facts = {}

    def replace(match):
        token = f'[[F{len(facts)}]]'
        facts[token] = match.group()
        return token
    masked = PROSE_MASK.sub(replace, "2026-yil II choragida narx 20,37 mln so'm.")
    assert masked == "[[F0]] narx [[F1]] mln so'm."
    assert list(facts.values()) == ['2026-yil II choragida', '20,37']
    assert '[[F[[' not in masked


def test_an_undecodable_reference_never_reaches_the_page():
    sections = content()
    llm = fake()
    llm.complete_json.side_effect = None
    # Whatever produced it, a paragraph that still carries a reference fragment
    # fails the run instead of being printed.
    llm.complete_json.return_value = {'blocks': [
        {'id': 's0b0', 'items': ['Narx [[F oshdi.']},
        {'id': 's1b0', 'items': ['Matn.']}]}
    with pytest.raises(LLMUnavailable):
        enrich(sections, llm)
    assert sections[0].blocks[0].items == ['Median 500 USD.']
