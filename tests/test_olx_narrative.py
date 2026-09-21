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
