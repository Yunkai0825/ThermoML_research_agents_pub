"""
Search & smart-query Flask Blueprint.

The ``/search`` route with its SQL query builder, similarity search
integration, and smart-query resolution-chip helpers.
"""

import traceback
import copy
import secrets
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

from flask import Blueprint, abort, current_app, render_template, request, url_for

from card_db_search_tools.basic_search_tools.normalization_helpers.strict_id_inputs import (
    IdentifierRefinementError,
    validate_search_values,
)

search_bp = Blueprint('search', __name__)


# ---------------------------------------------------------------------------
# Lazy-loaded similarity search
# ---------------------------------------------------------------------------

_sim_search = None


def _get_sim_search():
    global _sim_search
    if _sim_search is None:
        import importlib
        mod = importlib.import_module(
            "card_db_search_tools.basic_search_tools.10_compound_similarity_search"
        )
        _sim_search = mod.search_similar_compounds
    return _sim_search


# ---------------------------------------------------------------------------
# Helper: collect query params from request
# ---------------------------------------------------------------------------

def _search_params():
    """Collect all search params into a dict for template rendering."""
    return {
        'smart_q': request.args.get('smart_q', '').strip(),
        'resolved_search': request.args.get('resolved_search', '').strip(),
        'agentic': request.args.get('agentic', '').strip(),
        'agentic_limit': request.args.get('agentic_limit', '20').strip(),
        'doi': request.args.get('doi', '').strip(),
        'authors': request.args.get('authors', '').strip(),
        'title': request.args.get('title', '').strip(),
        'journal': request.args.get('journal', '').strip(),
        'year_min': request.args.get('year_min', '').strip(),
        'year_max': request.args.get('year_max', '').strip(),
        'compounds': request.args.get('compounds', '').strip(),
        'formula': request.args.get('formula', '').strip(),
        'system_type': request.args.get('system_type', '').strip(),
        'n_components': request.args.get('n_components', '').strip(),
        'sim_query': request.args.get('sim_query', '').strip(),
        'sim_threshold': request.args.get('sim_threshold', '').strip(),
        'sim_topk': request.args.get('sim_topk', '').strip(),
        'property': request.args.get('property', '').strip(),
        'property_kw': request.args.get('property_kw', '').strip(),
        'phase': request.args.get('phase', '').strip(),
        'block_type': request.args.get('block_type', '').strip(),
        'var_kw': request.args.get('var_kw', '').strip(),
        'variable': request.args.get('variable', '').strip(),
        'meas_kw': request.args.get('meas_kw', '').strip(),
        'measurement': request.args.get('measurement', '').strip(),
        'constr_kw': request.args.get('constr_kw', '').strip(),
        'constraint': request.args.get('constraint', '').strip(),
        'min_datapoints': request.args.get('min_datapoints', '').strip(),
        'max_wait_seconds': request.args.get('max_wait_seconds', '').strip(),
        'allow_query_agent': request.args.get('allow_query_agent', '').strip(),
        'agentic_all_fields': request.args.get('agentic_all_fields', '').strip(),
    }


# ---------------------------------------------------------------------------
# SQL query builder
# ---------------------------------------------------------------------------

def _build_search_query(q, alternatives=None):
    """Build SQL WHERE clause from search params.

    Returns (select_sql, count_sql, params).
    """
    joins = []
    wheres = []
    params = []

    # -- Bibliography --
    if q['doi']:
        wheres.append("r.doi LIKE ?")
        params.append(f"%{q['doi']}%")
    if q['authors']:
        wheres.append("r.first_author LIKE ?")
        params.append(f"%{q['authors']}%")
    if q['title']:
        for word in q['title'].split():
            wheres.append("r.title LIKE ?")
            params.append(f"%{word}%")
    if q['journal']:
        wheres.append("r.journal = ?")
        params.append(q['journal'])
    if q['year_min']:
        wheres.append("r.year >= ?")
        params.append(int(q['year_min']))
    if q['year_max']:
        wheres.append("r.year <= ?")
        params.append(int(q['year_max']))

    # -- Chemistry --
    if q['compounds']:
        compounds = validate_search_values(
            "compound", [v.strip() for v in q['compounds'].split(';') if v.strip()], field="compounds"
        )
        for i, compound in enumerate(compounds):
            alias = f"bc_{i}"
            joins.append(
                f"JOIN block_compounds {alias} ON {alias}.doi = bi.doi "
                f"AND {alias}.block_number = bi.block_number AND {alias}.block_type = bi.block_type"
            )
            if compound.startswith("GLOBcomp_"):
                wheres.append(f"{alias}.comp_num_id = ?")
                params.append(compound)
            else:
                wheres.append(f"{alias}.comp_name LIKE ?")
                params.append(f"%{compound.lower()}%")

    if q['formula']:
        joins.append(
            "JOIN block_compounds bcf ON bcf.doi = bi.doi "
            "AND bcf.block_number = bi.block_number AND bcf.block_type = bi.block_type"
        )
        wheres.append("bcf.comp_formula LIKE ?")
        params.append(f"%{q['formula']}%")

    # -- Properties / Phase --
    if q['property']:
        joins.append(
            "JOIN block_properties bprop ON bprop.doi = bi.doi "
            "AND bprop.block_number = bi.block_number AND bprop.block_type = bi.block_type"
        )
        property_atoms = validate_search_values(
            "property", q['property'], field="property"
        )
        if len(property_atoms) != 1:
            raise ValueError("property accepts exactly one property search value")
        if property_atoms[0].startswith("GLOBprop_"):
            wheres.append("bprop.prop_num_id = ?")
            params.append(property_atoms[0])
        else:
            wheres.append("bprop.prop_name LIKE ?")
            params.append(f'%{property_atoms[0]}%')
    if q['property_kw']:
        for index, kw in enumerate(validate_search_values("property", [v.strip() for v in q['property_kw'].split(';') if v.strip()], field="property_kw")):
            alias = f"bpkw_{index}"
            joins.append(
                f"JOIN block_properties {alias} ON {alias}.doi = bi.doi "
                f"AND {alias}.block_number = bi.block_number AND {alias}.block_type = bi.block_type"
            )
            if kw.startswith("GLOBprop_"):
                wheres.append(f"{alias}.prop_num_id = ?")
                params.append(kw)
            else:
                wheres.append(f"{alias}.prop_name LIKE ?")
                params.append(f'%{kw}%')

    if q['phase']:
        phase_atoms = validate_search_values("phase", [v.strip() for v in q['phase'].split(';') if v.strip()], field="phase")
        for phase in phase_atoms:
            if phase.startswith("GLOBphase_"):
                phase_clause = "bp2.phase_num_id = ?"
                params.append(phase)
            else:
                phase_clause = "bp2.phase_id LIKE ?"
                params.append(f'%{phase}%')
            wheres.append(
                "EXISTS (SELECT 1 FROM block_properties bp2 WHERE bp2.doi = bi.doi "
                "AND bp2.block_number = bi.block_number AND bp2.block_type = bi.block_type AND " + phase_clause + ")"
            )
    if q['block_type']:
        wheres.append("bi.block_type = ?")
        params.append(q['block_type'])

    # -- Variables --
    if q['variable']:
        joins.append(
            "JOIN block_variables bvar ON bvar.doi = bi.doi "
            "AND bvar.block_number = bi.block_number AND bvar.block_type = bi.block_type"
        )
        variable_atoms = validate_search_values(
            "variable", q['variable'], field="variable"
        )
        if len(variable_atoms) != 1:
            raise ValueError("variable accepts exactly one variable search value")
        if variable_atoms[0].startswith("GLOBvar_"):
            wheres.append("bvar.var_num_id = ?")
            params.append(variable_atoms[0])
        else:
            wheres.append("bvar.var_name LIKE ?")
            params.append(f'%{variable_atoms[0]}%')
    if q['var_kw']:
        for index, kw in enumerate(validate_search_values("variable", [v.strip() for v in q['var_kw'].split(';') if v.strip()], field="var_kw")):
            alias = f"bvkw_{index}"
            joins.append(
                f"JOIN block_variables {alias} ON {alias}.doi = bi.doi "
                f"AND {alias}.block_number = bi.block_number AND {alias}.block_type = bi.block_type"
            )
            if kw.startswith("GLOBvar_"):
                wheres.append(f"{alias}.var_num_id = ?")
                params.append(kw)
            else:
                wheres.append(f"{alias}.var_name LIKE ?")
                params.append(f'%{kw}%')

    # -- Measurements --
    if q['measurement']:
        joins.append(
            "JOIN block_measurements bmeas ON bmeas.doi = bi.doi "
            "AND bmeas.block_number = bi.block_number AND bmeas.block_type = bi.block_type"
        )
        measurement_atoms = validate_search_values(
            "measurement", q['measurement'], field="measurement"
        )
        if len(measurement_atoms) != 1:
            raise ValueError("measurement accepts exactly one measurement search value")
        if measurement_atoms[0].startswith("GLOBmeas_"):
            wheres.append("bmeas.meas_num_id = ?")
            params.append(measurement_atoms[0])
        else:
            wheres.append("bmeas.method_name LIKE ?")
            params.append(f'%{measurement_atoms[0]}%')
    if q['meas_kw']:
        for index, kw in enumerate(validate_search_values("measurement", [v.strip() for v in q['meas_kw'].split(';') if v.strip()], field="meas_kw")):
            alias = f"bmkw_{index}"
            joins.append(
                f"JOIN block_measurements {alias} ON {alias}.doi = bi.doi "
                f"AND {alias}.block_number = bi.block_number AND {alias}.block_type = bi.block_type"
            )
            if kw.startswith("GLOBmeas_"):
                wheres.append(f"{alias}.meas_num_id = ?")
                params.append(kw)
            else:
                wheres.append(f"{alias}.method_name LIKE ?")
                params.append(f'%{kw}%')

    # -- Constraints --
    if q['constraint']:
        joins.append(
            "JOIN block_constraints bcon ON bcon.doi = bi.doi "
            "AND bcon.block_number = bi.block_number AND bcon.block_type = bi.block_type"
        )
        constraint_atoms = validate_search_values(
            "constraint", q['constraint'], field="constraint"
        )
        if len(constraint_atoms) != 1:
            raise ValueError("constraint accepts exactly one constraint search value")
        if constraint_atoms[0].startswith("GLOBconstr_"):
            wheres.append("bcon.constr_num_id = ?")
            params.append(constraint_atoms[0])
        else:
            wheres.append("bcon.constr_name LIKE ?")
            params.append(f'%{constraint_atoms[0]}%')
    if q['constr_kw']:
        for index, kw in enumerate(validate_search_values("constraint", [v.strip() for v in q['constr_kw'].split(';') if v.strip()], field="constr_kw")):
            alias = f"bckw_{index}"
            joins.append(
                f"JOIN block_constraints {alias} ON {alias}.doi = bi.doi "
                f"AND {alias}.block_number = bi.block_number AND {alias}.block_type = bi.block_type"
            )
            if kw.startswith("GLOBconstr_"):
                wheres.append(f"{alias}.constr_num_id = ?")
                params.append(kw)
            else:
                wheres.append(f"{alias}.constr_name LIKE ?")
                params.append(f'%{kw}%')

    # System type / n_components
    if q['system_type']:
        wheres.append("bi.system_type = ?")
        params.append(q['system_type'])
    if q['n_components']:
        wheres.append("bi.n_components = ?")
        params.append(int(q['n_components']))

    if q['min_datapoints']:
        wheres.append("bi.n_datapoints >= ?")
        params.append(int(q['min_datapoints']))

    if alternatives is not None:
        if not alternatives:
            raise ValueError('No resolved OR groups are available. Submit the smart search again.')
        union = []
        for group in alternatives:
            group_sql, _, group_params = _build_search_query(group)
            union.append('SELECT doi, block_number, block_type FROM (' + group_sql + ')')
            params.extend(group_params)
        wheres.append('(bi.doi, bi.block_number, bi.block_type) IN (' + ' UNION '.join(union) + ')')

    # Assemble
    base_cols = (
        "bi.doi, bi.block_number, bi.block_type, bi.compound_system, "
        "bi.n_datapoints, bi.system_type, bi.n_components, r.doi as rdoi"
    )
    base = f"SELECT DISTINCT {base_cols} FROM block_index bi JOIN ref_index r ON r.doi = bi.doi"
    for j in joins:
        base += f" {j}"
    if wheres:
        base += " WHERE " + " AND ".join(wheres)

    count_sql = (
        "SELECT DISTINCT bi.doi, bi.block_number, bi.block_type, bi.n_datapoints "
        "FROM block_index bi JOIN ref_index r ON r.doi = bi.doi"
    )
    for j in joins:
        count_sql += f" {j}"
    if wheres:
        count_sql += " WHERE " + " AND ".join(wheres)

    # A block may have several matching properties, variables, or constraints.
    # Count its points once, matching the distinct block rows shown below.
    count_sql = ("SELECT COUNT(*), COUNT(DISTINCT doi), COALESCE(SUM(n_datapoints),0) "
                 "FROM (" + count_sql + ") matched_blocks")
    return base, count_sql, params


# ---------------------------------------------------------------------------
# Registry list helpers (used by search template dropdowns)
# ---------------------------------------------------------------------------

def _get_property_list(db):
    return db.execute(
        "SELECT prop_name, n_blocks FROM prop_registry ORDER BY n_blocks DESC"
    ).fetchall()


def _get_phase_list(db):
    return [r['phase_name'] for r in db.execute(
        "SELECT phase_name, n_occurrences FROM phase_registry ORDER BY n_occurrences DESC"
    ).fetchall()]


def _get_variable_list(db):
    return db.execute(
        "SELECT var_name, n_blocks FROM var_registry ORDER BY n_blocks DESC"
    ).fetchall()


def _get_measurement_list(db):
    return db.execute(
        "SELECT method_name, n_blocks FROM meas_registry ORDER BY n_blocks DESC"
    ).fetchall()


def _get_constraint_list(db):
    return db.execute(
        "SELECT constr_name, n_blocks FROM constr_registry ORDER BY n_blocks DESC"
    ).fetchall()


# ---------------------------------------------------------------------------
# Resolution chips (smart search feedback)
# ---------------------------------------------------------------------------

def _build_resolution_chips(rq):
    """Convert a ResolvedQuery into chip dicts for the template."""
    chips = []
    for r in rq.compounds:
        chips.append({'type': 'compound', 'type_label': 'Compound',
                      'text': r.display_name, 'global_id': r.global_id, 'score': r.score})
    for r in rq.properties:
        chips.append({'type': 'property', 'type_label': 'Property',
                      'text': r.display_name, 'global_id': r.global_id, 'score': r.score})
    for r in rq.measurements:
        chips.append({'type': 'measurement', 'type_label': 'Method',
                      'text': r.display_name, 'global_id': r.global_id, 'score': r.score})
    for r in rq.variables:
        chips.append({'type': 'variable', 'type_label': 'Variable',
                      'text': r.display_name, 'global_id': r.global_id, 'score': r.score})
    for r in rq.constraints:
        chips.append({'type': 'constraint', 'type_label': 'Constraint',
                      'text': r.display_name, 'global_id': r.global_id, 'score': r.score})
    for d in rq.dois:
        chips.append({'type': 'bibliography', 'type_label': 'DOI',
                      'text': d, 'global_id': None, 'score': None})
    for w in rq.title_words:
        chips.append({'type': 'bibliography', 'type_label': 'Title',
                      'text': w, 'global_id': None, 'score': None})
    for a in rq.authors:
        chips.append({'type': 'bibliography', 'type_label': 'Author',
                      'text': a, 'global_id': None, 'score': None})
    if rq.year_min:
        txt = str(rq.year_min)
        if rq.year_max:
            txt += f'\u2013{rq.year_max}'
        chips.append({'type': 'filter', 'type_label': 'Year',
                      'text': txt, 'global_id': None, 'score': None})
    if rq.system_type:
        chips.append({'type': 'filter', 'type_label': 'System',
                      'text': rq.system_type, 'global_id': None, 'score': None})
    if rq.block_type:
        chips.append({'type': 'filter', 'type_label': 'Block',
                      'text': rq.block_type, 'global_id': None, 'score': None})
    if rq.n_components:
        chips.append({'type': 'filter', 'type_label': '#Comp',
                      'text': str(rq.n_components), 'global_id': None, 'score': None})
    for f in rq.formulas:
        chips.append({'type': 'compound', 'type_label': 'Formula',
                      'text': f, 'global_id': None, 'score': None})
    return {
        'chips': chips,
        'unresolved': rq.unresolved,
        'log': rq.resolution_log,
    }


def _build_argo_resolution_chips(fld, argo_result):
    """Convert Argo agent resolved fields into chip dicts."""
    chips = []
    _ENTITY_MAP = {
        'compounds': ('compound', 'Compound'),
        'properties': ('property', 'Property'),
        'measurements': ('measurement', 'Method'),
        'variables': ('variable', 'Variable'),
        'constraints': ('constraint', 'Constraint'),
        'phases': ('phase', 'Phase'),
    }
    for key, (typ, label) in _ENTITY_MAP.items():
        for e in fld.get(key, []):
            chips.append({
                'type': typ, 'type_label': label,
                'text': e.get('name', ''), 'global_id': e.get('global_id'),
                'score': e.get('score'),
            })
    if fld.get('doi'):
        chips.append({'type': 'bibliography', 'type_label': 'DOI',
                      'text': fld['doi'], 'global_id': None, 'score': None})
    if fld.get('title_keywords'):
        chips.append({'type': 'bibliography', 'type_label': 'Title',
                      'text': fld['title_keywords'], 'global_id': None, 'score': None})
    if fld.get('authors'):
        chips.append({'type': 'bibliography', 'type_label': 'Author',
                      'text': fld['authors'], 'global_id': None, 'score': None})
    if fld.get('year_min'):
        txt = str(fld['year_min'])
        if fld.get('year_max'):
            txt += f'\u2013{fld["year_max"]}'
        chips.append({'type': 'filter', 'type_label': 'Year',
                      'text': txt, 'global_id': None, 'score': None})
    if fld.get('system_type'):
        chips.append({'type': 'filter', 'type_label': 'System',
                      'text': fld['system_type'], 'global_id': None, 'score': None})
    if fld.get('block_type'):
        chips.append({'type': 'filter', 'type_label': 'Block',
                      'text': fld['block_type'], 'global_id': None, 'score': None})
    if fld.get('formula'):
        chips.append({'type': 'compound', 'type_label': 'Formula',
                      'text': fld['formula'], 'global_id': None, 'score': None})
    return {
        'chips': chips,
        'unresolved': fld.get('unresolved', []),
        'log': fld.get('resolution_log', []),
        'argo_iterations': argo_result.iterations,
        'argo_elapsed': round(argo_result.elapsed_seconds, 1),
    }


def _build_hardcoded_resolution_chips(fld, result):
    """Convert hardcoded pipeline fields into chip dicts."""
    chips_data = _build_argo_resolution_chips(fld, result)
    chips_data.pop('argo_iterations', None)
    chips_data.pop('argo_elapsed', None)
    chips_data['deterministic'] = True
    chips_data['elapsed'] = round(result.elapsed_seconds, 3)
    chips_data['log'] = result.resolution_log or []
    return chips_data


# ---------------------------------------------------------------------------
# Route: Advanced Search
# ---------------------------------------------------------------------------

def _resolve_dynamic_name(name: str, component_name: str | None) -> str:
    """Append compound name to a dynamic entity name."""
    if component_name:
        return f"{name} ({component_name})"
    return name


_SEARCH_OPTIONS = frozenset({'sim_query', 'sim_threshold', 'sim_topk', 'smart_q', 'resolved_search',
                             'agentic', 'agentic_limit', 'max_wait_seconds',
                             'allow_query_agent', 'agentic_all_fields'})
_RESOLUTION_TTL = 1800
_RESOLUTION_LIMIT = 128


def _has_search_filter(q):
    return any(value for key, value in q.items() if key not in _SEARCH_OPTIONS)


def _merge_resolved_fields(q, fld):
    # Merge resolved fields into q (only fill empty fields)
    if fld.get('compounds') and not q['compounds']:
        q['compounds'] = '; '.join(c['global_id'] for c in fld['compounds'])
    if fld.get('properties') and not q['property_kw']:
        q['property_kw'] = '; '.join(p['global_id'] for p in fld['properties'])
    if fld.get('phases') and not q['phase']:
        q['phase'] = '; '.join(p['global_id'] for p in fld['phases'])
    if fld.get('title_keywords') and not q['title']:
        q['title'] = fld['title_keywords']
    if fld.get('doi') and not q['doi']:
        q['doi'] = fld['doi']
    if fld.get('authors') and not q['authors']:
        q['authors'] = fld['authors']
    if fld.get('journal') and not q['journal']:
        q['journal'] = fld['journal']
    if fld.get('year_min') and not q['year_min']:
        q['year_min'] = str(fld['year_min'])
    if fld.get('year_max') and not q['year_max']:
        q['year_max'] = str(fld['year_max'])
    if fld.get('system_type') and not q['system_type']:
        q['system_type'] = fld['system_type']
    if fld.get('block_type') and not q['block_type']:
        q['block_type'] = fld['block_type']
    if fld.get('n_components') and not q['n_components']:
        q['n_components'] = str(fld['n_components'])
    if fld.get('min_datapoints') and not q['min_datapoints']:
        q['min_datapoints'] = str(fld['min_datapoints'])
    if fld.get('formula') and not q['formula']:
        q['formula'] = fld['formula']
    if fld.get('variables') and not q['var_kw']:
        q['var_kw'] = '; '.join(v['global_id'] for v in fld['variables'])
    if fld.get('measurements') and not q['meas_kw']:
        q['meas_kw'] = '; '.join(m['global_id'] for m in fld['measurements'])
    if fld.get('constraints') and not q['constr_kw']:
        q['constr_kw'] = '; '.join(c['global_id'] for c in fld['constraints'])
    return q


def _search_resolution_cache():
    return current_app.extensions.setdefault('thermoml_search_resolutions',
                                             {'lock': threading.Lock(), 'entries': OrderedDict()})


def _clear_generated_search_fields(q, signature):
    """Discard only unchanged autofill values when the user changes smart input."""
    cache = _search_resolution_cache()
    with cache['lock']:
        item = cache['entries'].get(q['resolved_search'])
        if item and item['signature'] != signature and not request.args.get('page'):
            for key, value in item.get('generated_fields', {}).items():
                if q.get(key) == value:
                    q[key] = ''


def _get_search_resolution(token, signature):
    if not token:
        return None
    cache = _search_resolution_cache()
    with cache['lock']:
        item = cache['entries'].get(token)
        if item is None or time.monotonic() - item['created'] >= _RESOLUTION_TTL:
            cache['entries'].pop(token, None)
            if request.args.get('page'):
                raise ValueError('Saved search results expired. Use Restart smart search to resolve them again.')
            return None  # An explicit form submission can start a fresh resolution.
        if item['signature'] != signature:
            if request.args.get('page'):
                raise ValueError('Search text or settings changed. Use Restart smart search.')
            return None
        return copy.deepcopy(item['result'])


def _save_search_resolution(signature, result, generated_fields):
    cache = _search_resolution_cache()
    token = secrets.token_urlsafe(24)
    now = time.monotonic()
    with cache['lock']:
        entries = cache['entries']
        for key in list(entries):
            if now - entries[key]['created'] >= _RESOLUTION_TTL:
                entries.pop(key)
        while len(entries) >= _RESOLUTION_LIMIT:
            entries.popitem(last=False)
        entries[token] = {'created': now, 'signature': copy.deepcopy(signature), 'result': copy.deepcopy(result),
                          'generated_fields': copy.deepcopy(generated_fields)}
    return token


def _resolution_snapshot(result):
    return SimpleNamespace(**{key: copy.deepcopy(getattr(result, key, default)) for key, default in (
        ('fields', {}), ('query_groups', []), ('answer', ''), ('iterations', 0),
        ('elapsed_seconds', 0.0), ('timed_out', False), ('resolution_log', []))})


@search_bp.route('/search')
def search_page():
    # Resolve through the active mounted app. Importing a module named ``app``
    # is unsafe because another application may use that module name.
    db = current_app.extensions['thermoml_get_db']()
    q = _search_params()
    action = request.args.get('action', '')
    try:
        page = int(request.args.get('page', '1'))
        if page < 1:
            raise ValueError('page must be a positive integer')
        for field in ('year_min', 'year_max', 'n_components', 'min_datapoints'):
            if q[field] and int(q[field]) < 1:
                raise ValueError(f'{field} must be a positive integer')
    except ValueError as exc:
        abort(400, description=str(exc))

    # Dropdown data
    journals = [r[0] for r in db.execute(
        "SELECT DISTINCT journal FROM ref_index ORDER BY journal"
    ).fetchall()]
    top_properties = _get_property_list(db)
    phases = _get_phase_list(db)
    top_variables = _get_variable_list(db)
    top_measurements = _get_measurement_list(db)
    top_constraints = _get_constraint_list(db)

    # Stats for empty state
    stats = {
        'papers': db.execute("SELECT COUNT(*) FROM ref_index").fetchone()[0],
        'compounds': db.execute(
            "SELECT COUNT(*) FROM compound_registry"
        ).fetchone()[0],
        'blocks': db.execute("SELECT COUNT(*) FROM block_index").fetchone()[0],
    }

    # Structural similarity (independent of SQL search)
    sim_results = []
    if q['sim_query']:
        fn = _get_sim_search()
        if fn:
            try:
                threshold = float(q['sim_threshold'] or 0.5)
                topk = int(q['sim_topk'] or 20)
                queries = [s.strip() for s in q['sim_query'].split(';') if s.strip()]

                if len(queries) == 1:
                    res = fn(name=queries[0], top_k=topk, min_similarity=threshold)
                    if 'error' in res:
                        raise ValueError(res['error'])
                    for r in res['results']:
                        sim_results.append({
                            'name': r['common_name'],
                            'formula': r['formula'],
                            'smiles': r['smiles'],
                            'similarity': r['similarity'],
                            'n_papers': r['n_papers'],
                            'inchikey': r['inchi_key'],
                            'matched_query': queries[0],
                        })
                else:
                    combined = {}
                    for qname in queries:
                        per_q_k = max(topk * 2, 50)
                        res = fn(name=qname, top_k=per_q_k, min_similarity=0.0)
                        if 'error' in res:
                            raise ValueError(res['error'])
                        for r in res['results']:
                            cname = r['common_name']
                            ckey = (cname, r['formula'], r['smiles'])
                            if ckey not in combined:
                                combined[ckey] = {
                                    'scores': {},
                                    'info': {
                                        'name': cname,
                                        'formula': r['formula'],
                                        'smiles': r['smiles'],
                                        'n_papers': r['n_papers'],
                                        'inchikey': r['inchi_key'],
                                    },
                                }
                            prev = combined[ckey]['scores'].get(qname, 0)
                            combined[ckey]['scores'][qname] = max(prev, r['similarity'])

                    n_queries = len(queries)
                    scored = []
                    for ckey, entry in combined.items():
                        total = sum(entry['scores'].get(qn, 0) for qn in queries)
                        avg = total / n_queries
                        if avg < threshold:
                            continue
                        best_q = max(entry['scores'], key=entry['scores'].get)
                        scored.append({
                            **entry['info'],
                            'similarity': round(avg, 4),
                            'matched_query': best_q,
                            'per_query': {qn: round(entry['scores'].get(qn, 0), 4)
                                          for qn in queries},
                        })
                    scored.sort(key=lambda x: -x['similarity'])
                    sim_results = scored[:topk]
            except Exception:
                traceback.print_exc()
                raise

    # -- Smart search: parse free-text smart_q and merge into q --
    resolution_info = None
    resolution_groups = []
    alternatives = None
    search_error = None
    search_status = 200
    if q['smart_q'] and action:
        use_argo = q.get('agentic', '') in ('1', 'on', 'true')
        try:
            from ThermoML_results_browser.cannonical_id_alignment_search_agent.alignment_agent_api import (
                alignment_agent_run,
            )
            agentic_limit_text = q.get('agentic_limit', '20')
            if agentic_limit_text == "all":
                agentic_limit = 500
            elif agentic_limit_text.isdigit() and int(agentic_limit_text) > 0:
                agentic_limit = int(agentic_limit_text)
            else:
                raise ValueError("agentic_limit must be a positive integer or 'all'")
            max_wait = q.get('max_wait_seconds', '')
            allow_qagent = q.get('allow_query_agent', '') in ('1', 'on', 'true')
            agentic_all = q.get('agentic_all_fields', '') in ('1', 'on', 'true')
            align_settings: dict = {
                'use_agent': use_argo,
                'entry_limit': agentic_limit,
                'allow_query_agent': allow_qagent,
                'agentic_all_fields': agentic_all,
            }
            if max_wait:
                if not max_wait.isdigit() or int(max_wait) < 1:
                    raise ValueError('max_wait_seconds must be a positive integer')
                align_settings['max_wait_seconds'] = int(max_wait)

            signature = (q['smart_q'], align_settings)
            _clear_generated_search_fields(q, signature)
            generated_fields = {}
            result = _get_search_resolution(q['resolved_search'], signature)
            fresh = result is None
            if fresh:
                if request.args.get('page'):
                    raise ValueError('Saved search results are unavailable. Use Restart smart search.')
                try:
                    result = _resolution_snapshot(alignment_agent_run(q['smart_q'], settings=align_settings))
                except (ValueError, TimeoutError):
                    raise
                except Exception as exc:
                    raise RuntimeError('The alignment service could not complete the search.') from exc
            if result.timed_out:
                raise TimeoutError('Smart search reached its time limit')
            groups = result.query_groups
            if len(groups) > 1:
                alternatives = []
                for index, group in enumerate(groups, 1):
                    fld = group['fields']
                    group_q = _merge_resolved_fields({key: '' for key in q}, fld)
                    if fld.get('unresolved') or not _has_search_filter(group_q):
                        raise ValueError(f'OR group {index} could not be fully resolved. Refine that group and search again.')
                    alternatives.append(group_q)
                    details = SimpleNamespace(iterations=group.get('iterations', 0),
                                              elapsed_seconds=group.get('elapsed_seconds', 0.0),
                                              resolution_log=group.get('resolution_log', []))
                    chips = (_build_argo_resolution_chips(fld, details) if use_argo
                             else _build_hardcoded_resolution_chips(fld, details))
                    resolution_groups.append({'query': group['query'], 'details': chips})
                # Manual fields remain global filters; OR branches are never flattened into them.
            else:
                fld = result.fields
                resolution_info = (_build_argo_resolution_chips(fld, result) if use_argo
                                   else _build_hardcoded_resolution_chips(fld, result))
                before_merge = dict(q)
                _merge_resolved_fields(q, fld)
                generated_fields = {key: value for key, value in q.items() if value != before_merge[key]}
            if fresh:
                q['resolved_search'] = _save_search_resolution(signature, result, generated_fields)
        except ValueError as exc:
            search_error = str(exc)
            search_status = 400
        except TimeoutError:
            current_app.logger.warning('Smart search timed out')
            search_error = 'Smart search reached its time limit. Increase Wait or try again.'
            search_status = 504
        except Exception:
            current_app.logger.exception('Smart search failed')
            search_error = 'Smart search could not complete. Please try again.'
            search_status = 502

    # Check if any non-similarity filter is active
    has_filter = _has_search_filter(q) or alternatives is not None

    results = []
    total = n_papers = total_points = 0
    total_pages = 1
    count_only = False
    offset = 0

    if has_filter and action and not search_error:
        try:
            select_sql, count_sql, params = _build_search_query(q, alternatives)
        except ValueError as exc:
            abort(400, description=str(exc))

        row = db.execute(count_sql, params).fetchone()
        total, n_papers, total_points = row[0], row[1], row[2]

        if action == 'count':
            count_only = True
        else:
            per_page = 50
            offset = (page - 1) * per_page
            total_pages = max(1, (total + per_page - 1) // per_page)

            select_sql += f" ORDER BY bi.n_datapoints DESC, bi.doi, bi.block_type, bi.block_number LIMIT {per_page} OFFSET {offset}"
            rows = db.execute(select_sql, params).fetchall()

            for r in rows:
                comp_rows = db.execute(
                    "SELECT comp_name FROM block_compounds "
                    "WHERE doi = ? AND block_number = ? AND block_type = ?",
                    (r['doi'], r['block_number'], r['block_type']),
                ).fetchall()
                comp_names_list = [c['comp_name'] for c in comp_rows if c['comp_name']]

                prop_rows = db.execute(
                    "SELECT bp.prop_name, bp.component_org_num, bc.comp_name "
                    "FROM block_properties bp "
                    "LEFT JOIN block_compounds bc "
                    "ON bc.doi = bp.doi "
                    "AND bc.block_number = bp.block_number "
                    "AND bc.block_type = bp.block_type "
                    "AND bc.org_num = bp.component_org_num "
                    "WHERE bp.doi = ? AND bp.block_number = ? "
                    "AND bp.block_type = ?",
                    (r['doi'], r['block_number'], r['block_type']),
                ).fetchall()
                prop_names = [_resolve_dynamic_name(p['prop_name'], p['comp_name'])
                              for p in prop_rows]

                results.append({
                    'doi': r['doi'],
                    'block_number': r['block_number'],
                    'block_type': r['block_type'],
                    'compound_names': comp_names_list,
                    'property_names': prop_names,
                    'system_type': r['system_type'] or '',
                    'n_datapoints': r['n_datapoints'],
                })

    return render_template(
        'search.html',
        q=q,
        journals=journals,
        top_properties=top_properties,
        phases=phases,
        top_variables=top_variables,
        top_measurements=top_measurements,
        top_constraints=top_constraints,
        stats=stats,
        results=results,
        sim_results=sim_results,
        total=total,
        n_papers=n_papers,
        total_points=total_points,
        total_pages=total_pages,
        page=page,
        offset=offset,
        count_only=count_only,
        resolution_info=resolution_info,
        resolution_groups=resolution_groups,
        search_error=search_error,
        restart_url=url_for('search.search_page', **{key: value for key, value in q.items()
                                                   if key != 'resolved_search' and value}, action='search'),
    ), search_status
