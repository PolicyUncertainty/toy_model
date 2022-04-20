def utility(consumption, working_hours, hours_budget, gamma):
    out = consumption**gamma + (hours_budget - working_hours) ** (1 - gamma)
    return out


def hourly_wage_systematic(age, beta_0, beta_1, beta_2):
    out = beta_0 + beta_1 * age + beta_2 * age**2
    return out


def asset_update(
    assets_last_period,
    hourly_wage,
    working_hours,
    consumption,
    individual_long_term_costs,
    interest_rate,
):
    total_income = working_hours * hourly_wage
    compound_assets = (1 + interest_rate) * assets_last_period
    out = compound_assets + total_income - consumption - individual_long_term_costs
    return out


def calc_own_share_health(log_term_care_costs, insurance_share):
    out = (1 - insurance_share) * log_term_care_costs
    return out
