"""Reproduction commands, one module per reported result: python -m sen2fire_audit.reproduce <name>.
The computation and the order of random draws are those of the scripts behind the published numbers."""

cmd = {
    'supplement_tables': 'census and post hoc tables of the Supplement',
    'label_evidence_coupling': 'label-to-evidence coupling table',
    'seed_paired_contrasts': 'seed-paired and Welch analysis of the ten-seed comparisons',
    'sen2fire_deeplab_init_by_census_location': 'Sen2Fire DeepLabV3 initialization table',
    'cabuar_deeplab_init_contrasts': 'CaBuAr DeepLabV3 initialization table',
    'census_misplaced_reference': 'misplaced-label reference for the active-fire overlap',
    'active_fire_rule_sensitivity': 'active-fire rule variants against the census',
    'audit_rule_intervals': 'block bootstrap of the labelled-pixel spectral-rule rates',
    'audit_strata_intervals': 'block bootstrap of every stratum of the companion audit',
    'audit_strata_difference': 'non-fire minus labelled-fire burn-evidence rate',
    'block_size_sensitivity': 'block-size sensitivity of the spatial block bootstrap',
    'census_prevalence': 'census rates from the first author\'s blind answers',
    'census_agreement': 'agreement between the three annotators on the shared patches',
    'degradation_oracle_ceiling': 'degradation arms and their oracle ceiling',
    'degradation_gap_share': 'gap share G and learning ceiling of the degradation arms',
    'degradation_gap_matched': 'gap share with the matched Sen2Fire reference',
}
