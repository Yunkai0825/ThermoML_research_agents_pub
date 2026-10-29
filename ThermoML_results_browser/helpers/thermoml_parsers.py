"""
ThermoML JSON parsing helpers.

Pure functions that parse raw ThermoML JSON structures into display-ready
dicts. No Flask or filesystem dependency.

Raw ThermoML ordinal fields are used only while decoding the source document.
Every identifier exposed by this module uses the strict generated-card ID
grammar (``DOIcomp_*``, ``BLK*``, ``PROPblock_*``, or ``RXNblock_*``).
"""

import math
from decimal import Decimal

from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    block_id,
    block_local_id,
    doi_comp_id,
    doi_comp_sample_id,
)

# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def fmt_val(value):
    """Format a numeric value for clean display."""
    if value is None:
        return ''
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f'{value:.10g}'
    return str(value)


# ---------------------------------------------------------------------------
# Low-level element parsers
# ---------------------------------------------------------------------------

def _prop_group_info(prop_group):
    """Extract category, property name, and method from a PropertyGroup."""
    for key, val in prop_group.items():
        if key != 'tml_elements' and isinstance(val, dict):
            return {
                'category': key,
                'name': val.get('ePropName', ''),
                'method': val.get('eMethodName', val.get('sMethodName', '')),
            }
    return {'category': '', 'name': '', 'method': ''}


def _type_info(type_dict):
    """Return (type_key, label) from a VariableType / ConstraintType dict."""
    for key, val in type_dict.items():
        if key != 'tml_elements':
            return key, str(val)
    return '', ''


def _extract_uncertainty(prop_val):
    for cu in ensure_list(prop_val.get('CombinedUncertainty', [])):
        v = cu.get('nCombExpandUncertValue')
        if v is not None:
            return fmt_val(v)
        v = cu.get('nCombStdUncertValue')
        if v is not None:
            return fmt_val(v)
    return ''


# ThermoML ``ePresentation`` enum → short kind. Everything except 'direct' is a
# non-absolute scale (a difference or ratio against some reference state).
PRESENTATION_KINDS = {
    'Direct value, X': 'direct',
    'Difference with the reference state, X-X(REF)': 'diff',
    'Ratio with the reference state, X/X(REF)': 'ratio',
    'Ratio of difference with the reference state to the reference state, '
    '[X-X(REF)]/X(REF)': 'reldiff',
    'Difference between upper and lower temperature, X(T2)-X(T1)': 'diff_T',
    'Difference between upper and lower pressure, X(P2)-X(P1)': 'diff_P',
}

PRESENTATION_SHORT = {
    'direct': 'X',
    'diff': 'X-X(REF)',
    'ratio': 'X/X(REF)',
    'reldiff': '[X-X(REF)]/X(REF)',
    'diff_T': 'X(T2)-X(T1)',
    'diff_P': 'X(P2)-X(P1)',
}

# How the absolute value X is recovered from the stored value and X(REF).
ABSOLUTE_FORMULAS = {
    'diff': 'X = value + X(REF)',
    'ratio': 'X = value × X(REF)',
    'reldiff': 'X = X(REF) × (1 + value)',
}

# Short token spliced into the property label so a reader sees the scale at a
# glance, e.g. "Mass density, vs REF, kg/m3". Direct values carry no token.
PRESENTATION_TOKENS = {
    'diff': 'vs REF',
    'ratio': 'ratio vs REF',
    'reldiff': 'rel. diff vs REF',
    'diff_T': 'T2 vs T1',
    'diff_P': 'P2 vs P1',
}
ABSOLUTE_TOKEN = 'absolute'


def labelled_name(name, token):
    """Insert ``token`` before the unit: 'Mass density, kg/m3' → 'Mass density, vs REF, kg/m3'.

    ThermoML property names are '<name>, <unit>' or unit-less '<name>'; the
    unit never contains a space, so a trailing ', <no-space>' segment is the unit.
    """
    if not token:
        return name
    head, sep, tail = name.rpartition(', ')
    if sep and ' ' not in tail:
        return f'{head}, {token}, {tail}'
    return f'{name}, {token}'


def _presentation_kind(prop):
    pres = prop.get('ePresentation')
    if pres is None:
        return 'direct'
    return PRESENTATION_KINDS.get(pres, 'other')


def _parse_ref_state(prop, compounds_map):
    """Reference-state metadata of a Property, or None when the file has none.

    Raw fields: ``eRefStateType``, ``RefPhaseID{eRefPhase, RegNum}``,
    ``nRefTemp``/``nRefPressure`` (fixed reference conditions) and ``Solvent``
    (the reference compound for the 'Pure solvent …' reference types).
    """
    kind = _presentation_kind(prop)
    ref_phase = prop.get('RefPhaseID') or {}
    ref_type = prop.get('eRefStateType', '')
    has_ref_fields = bool(ref_phase) or bool(ref_type) or (
        prop.get('nRefTemp') is not None or prop.get('nRefPressure') is not None
    )
    if kind == 'direct' and not has_ref_fields:
        return None

    ref_org = ref_phase.get('RegNum', {}).get('nOrgNum') if 'RegNum' in ref_phase else None
    solvent_orgs = [
        r.get('nOrgNum')
        for r in ensure_list((prop.get('Solvent') or {}).get('RegNum', []))
        if r.get('nOrgNum') is not None
    ]
    compound = compounds_map.get(ref_org, {}).get('name', '') if ref_org is not None else ''
    solvents = [compounds_map.get(o, {}).get('name', f'compound {doi_comp_id(o)}') for o in solvent_orgs]
    phase = ref_phase.get('eRefPhase', '')
    temperature = fmt_val(prop.get('nRefTemp')) if prop.get('nRefTemp') is not None else None
    pressure = fmt_val(prop.get('nRefPressure')) if prop.get('nRefPressure') is not None else None

    # Human-readable parts, in display order; only what the file actually states.
    summary = []
    if ref_type:
        summary.append(ref_type)
    if phase:
        summary.append(f'Reference phase: {phase}')
    if compound:
        summary.append(f'Reference compound: {compound}')
        if solvents and solvents != [compound]:
            summary.append('Solvent: ' + ' + '.join(solvents))
    elif len(solvents) == 1 and 'solvent' in ref_type.lower():
        summary.append(f'Reference compound: {solvents[0]} (the solvent)')
    elif solvents:
        summary.append('Solvent: ' + ' + '.join(solvents))
    if temperature is not None:
        summary.append(f'T(REF) = {temperature} K')
    if pressure is not None:
        summary.append(f'P(REF) = {pressure} kPa')

    return {
        'type': ref_type,
        'phase': phase,
        'compound': compound,
        'compound_org_num': doi_comp_id(ref_org) if ref_org is not None else None,
        'solvents': solvents,
        'temperature': temperature,
        'pressure': pressure,
        'presentation_short': PRESENTATION_SHORT.get(kind, prop.get('ePresentation', '')),
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# Parse individual elements
# ---------------------------------------------------------------------------

def parse_property(prop, compounds_map=None):
    method_id = prop.get('Property-MethodID', {})
    gi = _prop_group_info(method_id.get('PropertyGroup', {}))

    uncert_meta = None
    for cu in ensure_list(prop.get('CombinedUncertainty', [])):
        uncert_meta = {
            'evaluator': cu.get('sCombUncertEvaluator', ''),
            'method': cu.get('eCombUncertEvalMethod', ''),
            'confidence': cu.get('nCombUncertLevOfConfid'),
        }

    kind = _presentation_kind(prop)
    return {
        'BLKprop_id': block_local_id('prop', prop.get('nPropNumber')),
        'name': gi['name'],
        'display_name': labelled_name(gi['name'], PRESENTATION_TOKENS.get(kind)),
        'category': gi['category'],
        'method': gi['method'],
        'phase': prop.get('PropPhaseID', {}).get('ePropPhase', ''),
        'presentation': prop.get('ePresentation', ''),
        'presentation_kind': kind,
        'ref_state': _parse_ref_state(prop, compounds_map or {}),
        'absolute': None,  # filled by attach_absolute_columns() at paper level
        'temperature': prop.get('nTemperature-K'),
        'pressure': prop.get('nPressure-kPa'),
        'uncertainty_meta': uncert_meta,
    }


def parse_variable(var, compounds_map):
    var_id = var.get('VariableID', {})
    type_key, type_label = _type_info(var_id.get('VariableType', {}))
    reg_num = var_id.get('RegNum', {}).get('nOrgNum') if 'RegNum' in var_id else None
    label = type_label
    if reg_num is not None and reg_num in compounds_map:
        label = f"{type_label} ({compounds_map[reg_num]['name']})"
    return {
        'BLKvar_id': block_local_id('var', var.get('nVarNumber')),
        'type_key': type_key,
        'label': label,
        'component_org_num': doi_comp_id(reg_num) if reg_num is not None else None,
        'phase': var.get('VarPhaseID', {}).get('eVarPhase', ''),
    }


def parse_constraint(con, compounds_map):
    con_id = con.get('ConstraintID', {})
    type_key, type_label = _type_info(con_id.get('ConstraintType', {}))
    reg_num = con_id.get('RegNum', {}).get('nOrgNum') if 'RegNum' in con_id else None
    label = type_label
    if reg_num is not None and reg_num in compounds_map:
        label = f"{type_label} ({compounds_map[reg_num]['name']})"
    return {
        'BLKconstr_id': block_local_id('constr', con.get('nConstraintNumber')),
        'label': label,
        'value': fmt_val(con.get('nConstraintValue')),
        'phase': con.get('ConstraintPhaseID', {}).get('eConstraintPhase', ''),
        'component_org_num': doi_comp_id(reg_num) if reg_num is not None else None,
    }


def parse_samples(samples_raw, org_ordinal):
    result = []
    for sample in ensure_list(samples_raw):
        sample_ordinal = sample.get('nSampleNm')
        s = {
            'sample_num': doi_comp_sample_id(org_ordinal, sample_ordinal),
            'source': sample.get('eSource', ''),
            'steps': [],
        }
        for pur in ensure_list(sample.get('purity', [])):
            # nStep is source sequence metadata, not an identifier.
            step = {'step_index': pur.get('nStep')}
            if 'nPurityMol' in pur:
                step['purity'] = f"{pur['nPurityMol']}% mol"
            elif 'nPurityMass' in pur:
                step['purity'] = f"{pur['nPurityMass']}% mass"
            elif 'nPurityVol' in pur:
                step['purity'] = f"{pur['nPurityVol']}% vol"
            else:
                step['purity'] = ''
            if 'nWaterMassPerCent' in pur:
                step['water'] = f"{pur['nWaterMassPerCent']}% water"
            else:
                step['water'] = ''
            step['methods'] = ensure_list(pur.get('ePurifMethod', []))
            step['analyses'] = ensure_list(pur.get('eAnalMeth', []))
            s['steps'].append(step)
        result.append(s)
    return result


# ---------------------------------------------------------------------------
# Parse full measurement blocks
# ---------------------------------------------------------------------------

def _build_data_table(block, variables, properties):
    """Build column definitions and row data from NumValues."""
    columns = []
    for v in variables:
        columns.append({
            'type': 'variable', 'local_id': v['BLKvar_id'], 'name': v['label'],
        })
    for p in properties:
        columns.append({
            'type': 'property', 'local_id': p['BLKprop_id'], 'name': p['name'],
        })
        columns.append({
            'type': 'uncertainty', 'local_id': p['BLKprop_id'], 'name': '±',
        })

    rows = []
    for nv in ensure_list(block.get('NumValues', [])):
        var_map = {
            block_local_id('var', vv.get('nVarNumber')): vv
            for vv in ensure_list(nv.get('VariableValue', []))
        }
        prop_map = {
            block_local_id('prop', pv.get('nPropNumber')): pv
            for pv in ensure_list(nv.get('PropertyValue', []))
        }
        row = []
        for col in columns:
            if col['type'] == 'variable':
                row.append(fmt_val(var_map.get(col['local_id'], {}).get('nVarValue')))
            elif col['type'] == 'property':
                row.append(fmt_val(prop_map.get(col['local_id'], {}).get('nPropValue')))
            elif col['type'] == 'uncertainty':
                row.append(_extract_uncertainty(prop_map.get(col['local_id'], {})))
        rows.append(row)

    return columns, rows


def _display_columns(columns, properties):
    """Browser copy of ``columns`` with representation tokens in property labels."""
    labels = {
        p['BLKprop_id']: p['display_name']
        for p in properties if p['display_name'] != p['name']
    }
    if not labels:
        return columns
    return [
        dict(col, name=labels[col['local_id']])
        if col['type'] == 'property' and col['local_id'] in labels else col
        for col in columns
    ]


def parse_pom_block(block, compounds_map):
    components = []
    for c in ensure_list(block.get('Component', [])):
        org = c.get('RegNum', {}).get('nOrgNum')
        ci = compounds_map.get(org, {})
        sample_ordinal = c.get('nSampleNm')
        components.append({
            'org_num': doi_comp_id(org),
            'sample_num': (
                doi_comp_sample_id(org, sample_ordinal)
                if sample_ordinal is not None else None
            ),
            'name': ci.get('name', ''),
            'formula': ci.get('formula', ''),
        })

    properties = [parse_property(p, compounds_map) for p in ensure_list(block.get('Property', []))]
    phases = [p.get('ePhase', '') for p in ensure_list(block.get('PhaseID', []))]
    constraints = [parse_constraint(c, compounds_map) for c in ensure_list(block.get('Constraint', []))]
    variables = [parse_variable(v, compounds_map) for v in ensure_list(block.get('Variable', []))]
    columns, rows = _build_data_table(block, variables, properties)

    return {
        'type': 'PureOrMixtureData',
        'block_number': block_id(
            'PureOrMixtureData', block.get('nPureOrMixtureDataNumber')
        ),
        'purpose': block.get('eExpPurpose', ''),
        'compiler': block.get('sCompiler', ''),
        'contributor': block.get('sContributor', ''),
        'date_added': block.get('dateDateAdded', ''),
        'components': components,
        'properties': properties,
        'phases': phases,
        'constraints': constraints,
        'variables': variables,
        'columns': columns,
        'rows': rows,
        # Browser-only view of the table: tokenised property labels, and possibly
        # derived columns (see attach_absolute_columns). ``columns``/``rows``
        # stay exactly as stored.
        'display_columns': _display_columns(columns, properties),
        'display_rows': rows,
        'n_datapoints': len(rows),
        'has_fixed_temp': any(p['temperature'] is not None for p in properties),
        'has_fixed_pres': any(p['pressure'] is not None for p in properties),
    }


def parse_reaction_block(block, compounds_map):
    participants = []
    for p in ensure_list(block.get('Participant', [])):
        org = p.get('RegNum', {}).get('nOrgNum')
        ci = compounds_map.get(org, {})
        sample_ordinal = p.get('nSampleNm')
        participants.append({
            'org_num': doi_comp_id(org),
            'name': ci.get('name', ''),
            'formula': ci.get('formula', ''),
            # text in the raw corpus ("-1", ".5", "-10.25"): a finite number, else 0 (finding (g), 2026-10-21)
            'stoichiometry': _stoichiometric_coefficient(p.get('nStoichiometricCoef')),
            'phase': p.get('ePhase', ''),
            'sample_num': (
                doi_comp_sample_id(org, sample_ordinal)
                if sample_ordinal is not None else None
            ),
        })

    # formatted equation
    reactants = [p for p in participants if p['stoichiometry'] < 0]
    products = [p for p in participants if p['stoichiometry'] > 0]

    def _fmt_p(p):
        c = abs(p['stoichiometry'])
        cs = f"{fmt_val(c)} " if c != 1 else ""      # 2.0 -> "2 ", 2.5 -> "2.5 ", 1 -> nothing
        return f"{cs}{p['name']} ({p['phase']})"

    equation = ' + '.join(_fmt_p(r) for r in reactants) + ' \u2192 ' + ' + '.join(_fmt_p(pr) for pr in products)

    properties = [parse_property(p, compounds_map) for p in ensure_list(block.get('Property', []))]
    variables = []  # reactions rarely have variables
    columns, rows = _build_data_table(block, variables, properties)

    return {
        'type': 'ReactionData',
        'block_number': block_id('ReactionData', block.get('nReactionDataNumber')),
        'purpose': block.get('eExpPurpose', ''),
        'compiler': block.get('sCompiler', ''),
        'contributor': block.get('sContributor', ''),
        'date_added': block.get('dateDateAdded', ''),
        'reaction_type': block.get('eReactionType', ''),
        'participants': participants,
        'equation': equation,
        'properties': properties,
        'phases': [],
        'constraints': [],
        'variables': variables,
        'columns': columns,
        'rows': rows,
        'display_columns': _display_columns(columns, properties),
        'display_rows': rows,
        'n_datapoints': len(rows),
        'has_fixed_temp': any(p['temperature'] is not None for p in properties),
        'has_fixed_pres': any(p['pressure'] is not None for p in properties),
    }


# ---------------------------------------------------------------------------
# Absolute-value reconstruction for non-absolute presentations
# ---------------------------------------------------------------------------
#
# ThermoML stores X-X(REF) / X/X(REF) / [X-X(REF)]/X(REF) values but never
# X(REF) itself.  The only DB-grounded source for X(REF) is another block of the
# SAME paper that reports the same property of the pure reference compound as a
# direct value; rows are joined on exactly coinciding (T, P).  Nothing external
# is ever used; when no such block exists the absence is reported, not guessed.

_T_KEY, _P_KEY = 'T', 'P'


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stoichiometric_coefficient(value):
    """A reaction participant's nStoichiometricCoef as a finite float.

    The raw corpus carries it as text ("-1", "1", ".5", "-10.25"); a missing, unreadable or
    non-finite value counts as 0.0 (the participant is then on neither side of the equation).
    """
    coef = _as_float(value)
    return coef if coef and math.isfinite(coef) else 0.0


def _state_axes(raw_block):
    """(fixed {T/P: value}, {nVarNumber: 'T'|'P'}) from constraints/variables."""
    fixed = {}
    for con in ensure_list(raw_block.get('Constraint', [])):
        key, label = _type_info(con.get('ConstraintID', {}).get('ConstraintType', {}))
        axis = _axis_for(key, label)
        if axis and _as_float(con.get('nConstraintValue')) is not None:
            fixed[axis] = _as_float(con['nConstraintValue'])
    var_axes = {}
    for var in ensure_list(raw_block.get('Variable', [])):
        key, label = _type_info(var.get('VariableID', {}).get('VariableType', {}))
        axis = _axis_for(key, label)
        if axis:
            var_axes[var.get('nVarNumber')] = axis
    return fixed, var_axes


def _axis_for(type_key, type_label):
    if type_key == 'eTemperature' and type_label == 'Temperature, K':
        return _T_KEY
    if type_key == 'ePressure' and type_label == 'Pressure, kPa':
        return _P_KEY
    return None


def _state_points(raw_block, prop_ordinal):
    """Per NumValues row: ((T, P), raw property value) for one property."""
    fixed, var_axes = _state_axes(raw_block)
    points = []
    for nv in ensure_list(raw_block.get('NumValues', [])):
        state = dict(fixed)
        for vv in ensure_list(nv.get('VariableValue', [])):
            axis = var_axes.get(vv.get('nVarNumber'))
            if axis and _as_float(vv.get('nVarValue')) is not None:
                state[axis] = _as_float(vv['nVarValue'])
        value = None
        for pv in ensure_list(nv.get('PropertyValue', [])):
            if pv.get('nPropNumber') == prop_ordinal:
                value = pv.get('nPropValue')
        points.append(((state.get(_T_KEY), state.get(_P_KEY)), value))
    return points


def _decimals(text):
    exponent = Decimal(text).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _sigfigs(text):
    return max(1, len(Decimal(text).as_tuple().digits))


def _combine(kind, value, ref):
    """Absolute value as a display string, precision limited by the coarser operand.

    Returns '' when either operand is not numeric.
    """
    v_txt, r_txt = fmt_val(value), fmt_val(ref)
    try:
        v, r = float(v_txt), float(r_txt)
        if kind == 'diff':
            return fmt_val(round(v + r, min(_decimals(v_txt), _decimals(r_txt))))
        x = v * r if kind == 'ratio' else r * (1.0 + v)
        sig = min(_sigfigs(v_txt), _sigfigs(r_txt))
        return fmt_val(float(f'{x:.{sig}g}'))
    except (ValueError, ArithmeticError):
        return ''


def _reference_org(prop):
    """Raw ordinal of the pure reference compound, or None when not a single compound."""
    ref_phase = prop.get('RefPhaseID') or {}
    if 'RegNum' in ref_phase and ref_phase['RegNum'].get('nOrgNum') is not None:
        return ref_phase['RegNum']['nOrgNum']
    solvents = [
        r.get('nOrgNum')
        for r in ensure_list((prop.get('Solvent') or {}).get('RegNum', []))
    ]
    if len(solvents) == 1 and solvents[0] is not None:
        return solvents[0]
    return None


def attach_absolute_columns(raw_poms, measurements, compounds_map):
    """Add derived absolute-value columns to ``display_columns``/``display_rows``.

    ``raw_poms`` and ``measurements`` are parallel lists (raw block, parsed block).
    Each non-direct property receives ``absolute`` = {'status': 'derived', ...}
    or {'status': 'unavailable', 'reason': ...}; ``columns``/``rows`` are untouched.
    """
    if not any(p['presentation_kind'] != 'direct'
               for meas in measurements for p in meas['properties']):
        return

    # Index of pure-compound direct-value properties: (org, prop name) → sources
    pure_sources = {}
    for raw, meas in zip(raw_poms, measurements):
        comps = [c.get('RegNum', {}).get('nOrgNum') for c in ensure_list(raw.get('Component', []))]
        if len(comps) != 1:
            continue
        for prop in ensure_list(raw.get('Property', [])):
            if _presentation_kind(prop) != 'direct':
                continue
            name = _prop_group_info(prop.get('Property-MethodID', {}).get('PropertyGroup', {}))['name']
            pure_sources.setdefault((comps[0], name), []).append({
                'block_number': meas['block_number'],
                'phase': prop.get('PropPhaseID', {}).get('ePropPhase', ''),
                'points': _state_points(raw, prop.get('nPropNumber')),
            })

    for raw, meas in zip(raw_poms, measurements):
        extra_cols, extra_vals = [], []
        # parse_pom_block builds ``properties`` from Property in source order.
        for prop, parsed in zip(ensure_list(raw.get('Property', [])), meas['properties']):
            kind = parsed['presentation_kind']
            if kind == 'direct':
                continue
            parsed['absolute'] = _resolve_absolute(prop, parsed, kind, raw, pure_sources, compounds_map)
            if parsed['absolute']['status'] == 'derived':
                extra_cols.append({
                    'type': 'absolute',
                    'local_id': parsed['BLKprop_id'],
                    'name': labelled_name(parsed['name'], ABSOLUTE_TOKEN),
                    'source_block': parsed['absolute']['source_block'],
                })
                extra_vals.append(parsed['absolute'].pop('_values'))
        if extra_cols:
            meas['display_columns'] = list(meas['display_columns']) + extra_cols
            meas['display_rows'] = [
                list(row) + [vals[i] for vals in extra_vals]
                for i, row in enumerate(meas['rows'])
            ]


def _resolve_absolute(prop, parsed, kind, raw_block, pure_sources, compounds_map):
    if kind in ('diff_T', 'diff_P'):
        return {
            'status': 'unavailable',
            'reason': f"{PRESENTATION_SHORT[kind]} is a difference between two states; "
                      "neither end state is stored in this file.",
        }
    if kind not in ABSOLUTE_FORMULAS:
        return {
            'status': 'unavailable',
            'reason': f"presentation {parsed['presentation']!r} is not an absolute scale "
                      "and has no reconstruction rule.",
        }
    ref_org = _reference_org(prop)
    ref_state = parsed['ref_state'] or {}
    if ref_org is None:
        return {
            'status': 'unavailable',
            'reason': "the reference state is not a single pure compound"
                      + (f" ({ref_state['type']})" if ref_state.get('type') else '')
                      + "; X(REF) cannot be located in this file.",
        }
    ref_name = compounds_map.get(ref_org, {}).get('name', '') or f"compound {doi_comp_id(ref_org)}"
    sources = pure_sources.get((ref_org, parsed['name']), [])
    ref_phase = ref_state.get('phase', '')
    if ref_phase:
        sources = [s for s in sources if not s['phase'] or s['phase'] == ref_phase]
    if not sources:
        return {
            'status': 'unavailable',
            'reason': f"X(REF) = {parsed['name']} of pure {ref_name}"
                      f"{' (' + ref_phase + ')' if ref_phase else ''} at each (T, P) "
                      "is not stored in this file.",
        }

    # (T, P) → X(REF); ambiguous duplicates are dropped rather than picked.
    lookup = {}
    for src in sources:
        for state, value in src['points']:
            if value is not None:
                lookup.setdefault(state, set()).add(fmt_val(value))
    lookup = {k: next(iter(v)) for k, v in lookup.items() if len(v) == 1}

    fixed_T = _as_float(prop.get('nRefTemp'))
    fixed_P = _as_float(prop.get('nRefPressure'))
    values, matched = [], 0
    for (T, P), value in _state_points(raw_block, prop.get('nPropNumber')):
        key = (fixed_T if fixed_T is not None else T, fixed_P if fixed_P is not None else P)
        ref = lookup.get(key)
        absolute = _combine(kind, value, ref) if ref is not None and value is not None else ''
        values.append(absolute)
        matched += bool(absolute)
    blocks = sorted({s['block_number'] for s in sources})
    if not matched:
        return {
            'status': 'unavailable',
            'reason': f"pure {ref_name} {parsed['name']} exists in {', '.join(blocks)} "
                      "but at no (T, P) coinciding with this block's rows.",
        }
    return {
        'status': 'derived',
        'formula': ABSOLUTE_FORMULAS[kind],
        'source_block': ', '.join(blocks),
        'source_compound': ref_name,
        'n_matched': matched,
        'n_rows': len(values),
        '_values': values,
    }


def parse_paper_data(data: dict):
    """Fully parse one decoded ThermoML JSON document.

    Parameters
    ----------
    data : dict
        Raw ThermoML JSON structure loaded from the compressed corpus.
    """
    compounds_map = {}
    compounds_list = []
    for comp in ensure_list(data.get('Compound', [])):
        org = comp.get('RegNum', {}).get('nOrgNum')
        names = ensure_list(comp.get('sCommonName', []))
        info = {
            'org_num': doi_comp_id(org),
            'names': names,
            'name': names[0] if names else '',
            'formula': comp.get('sFormulaMolec', ''),
            'inchi': comp.get('sStandardInChI', ''),
            'inchikey': comp.get('sStandardInChIKey', ''),
            'samples': parse_samples(comp.get('Sample', []), org),
        }
        compounds_map[org] = info
        compounds_list.append(info)

    raw_poms = ensure_list(data.get('PureOrMixtureData', []))
    measurements = [parse_pom_block(pom, compounds_map) for pom in raw_poms]
    attach_absolute_columns(raw_poms, measurements, compounds_map)
    for rd in ensure_list(data.get('ReactionData', [])):
        measurements.append(parse_reaction_block(rd, compounds_map))

    total_pts = sum(m['n_datapoints'] for m in measurements)

    return {
        'compounds': compounds_list,
        'measurements': measurements,
        'total_datapoints': total_pts,
    }
