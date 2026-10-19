"""Parent clean evaluation plus exact number of quantum token evaluations."""
from experiment_metrics import evaluate as parent_evaluate
from replacement_model import Replacement

def evaluate(model, dataset, cache=None, draws=0, seed=0):
    modules = [m for m in model.modules() if isinstance(m, Replacement)]
    for m in modules:
        m.token_evaluations = 0
        m.collect_stats = True
    try:
        result = parent_evaluate(model, dataset, cache=None, draws=draws, seed=seed)
        result['quantum_token_evaluations'] = sum(m.token_evaluations for m in modules)
        result['quantum_token_evaluations_per_example'] = result['quantum_token_evaluations'] / len(dataset)
        return result
    finally:
        for m in modules:
            m.collect_stats = False
