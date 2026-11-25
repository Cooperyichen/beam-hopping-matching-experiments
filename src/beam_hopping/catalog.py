"""Small dependency-free discovery catalogue; implementations remain separate."""

STRATEGIES = (
    ('throughput_matching', 'Forecast comparison', 'synthetic_traffic_validation.allocation', 'matching_plan', 'Capacity only; lambda=0'),
    ('urgency_no_prediction', 'Forecast comparison', 'synthetic_traffic_validation.allocation', 'matching_plan', 'Historical mean demand target'),
    ('prediction_matching', 'Forecast comparison', 'synthetic_traffic_validation.allocation', 'matching_plan', 'Predicted demand target'),
    ('reserve_matching', 'Forecast comparison', 'synthetic_traffic_validation.allocation', 'matching_plan', 'Prediction plus reserve'),
    ('static_trust_matching', 'Forecast comparison', 'synthetic_traffic_validation.allocation', 'matching_plan', 'Trust weighted reserve'),
    ('certified_static_matching', 'Forecast comparison', 'synthetic_traffic_validation.revision_code.revision', 'hybrid_plan', 'Checks and optional repair'),
    ('nominal_horizon', 'Horizon optimization', 'synthetic_traffic_validation.allocation', 'solve_allocation', 'No channel allowance or demand reserve'),
    ('chance_horizon', 'Horizon optimization', 'synthetic_traffic_validation.allocation', 'solve_allocation', 'Model based service protections'),
    ('deficit_matching', 'Consecutive feedback', 'synthetic_traffic_validation.deficit_matching', 'DeficitMatchingController', 'Frozen cycle deficit weights'),
    ('single_slot_exact', 'Fixed resource matching', 'beam_hopping.matching.single_slot', 'solve_matching', 'MOSEK LP or binary benchmark'),
    ('greedy_throughput', 'Heuristic comparison', 'beam_hopping.matching.heuristics', 'greedy_matching', 'Greedy feasible edge selection'),
    ('demand_priority_greedy', 'Heuristic comparison', 'beam_hopping.matching.heuristics', 'demand_priority_greedy', 'Demand weighted greedy selection'),
    ('genetic_algorithm', 'Heuristic comparison', 'beam_hopping.matching.heuristics', 'genetic_matching', 'Population search baseline'),
    ('online_heuristics', 'Earlier prototype', 'beam_hopping.scheduling.online', 'run_policy', 'Separate synthetic horizon model'),
    ('single_agent_dqn', 'Learning prototype', 'beam_hopping.learning.reinforcement', 'train_single_agent_dqn', 'Queue based Deep Q Network'),
    ('cooperative_iql', 'Learning prototype', 'beam_hopping.learning.reinforcement', 'train_cooperative_iql', 'Cooperative independent Q learning'),
)


def entries():
    fields = ('name', 'family', 'module', 'entry', 'summary')
    return [dict(zip(fields, row)) for row in STRATEGIES]
