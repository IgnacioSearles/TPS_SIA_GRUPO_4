import copy
import json

from src.evaluation.art_search_cache import audited_legacy_cache


def test_only_exact_audited_source_transition_can_reuse_cache(tmp_path):
    stable = {'src/evaluation/art_geometry.py':'geometry', 'src/evaluation/art_cleanliness.py':'clean'}
    before = {**stable,'src/models/factory.py':'old-factory','src/nn/spatial.py':'old-spatial'}
    after = {**stable,'src/models/factory.py':'new-factory','src/nn/spatial.py':'new-spatial'}
    old = {'sources':[{'model':'weights'}],'data':'same','loss_guard':{'loss':'same'},'evaluation_sha256':before}
    new = {**old,'evaluation_sha256':after}
    assert not audited_legacy_cache(tmp_path,'pilot',old,new)
    path = tmp_path/'results/art_search/refinement_compatibility.json'
    path.parent.mkdir(parents=True)
    proof = {'loss_guard':old['loss_guard'],'current_evaluation_sha256':after,
             'legacy_predictions':[{'candidate':'pilot','decoded_outputs_bitwise_identical':True,
                                    'previous_evaluation_sha256':before}]}
    path.write_text(json.dumps(proof))
    assert audited_legacy_cache(tmp_path,'pilot',old,new)
    assert not audited_legacy_cache(tmp_path,'other',old,new)
    for field in ('data','sources','loss_guard','evaluation_sha256'):
        changed = copy.deepcopy(new); changed[field] = 'changed'
        assert not audited_legacy_cache(tmp_path,'pilot',old,changed)
    # Even a claimed proof cannot approve altered metrics.
    proof['current_evaluation_sha256']['src/evaluation/art_geometry.py']='different'
    path.write_text(json.dumps(proof))
    changed = copy.deepcopy(new); changed['evaluation_sha256']=proof['current_evaluation_sha256']
    assert not audited_legacy_cache(tmp_path,'pilot',old,changed)
