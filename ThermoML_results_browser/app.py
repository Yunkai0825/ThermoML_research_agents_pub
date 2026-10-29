#!/usr/bin/env python3
"""
NIST ThermoML Database Browser — Flask web application.

Usage:
    python launch_thermoml_browser.py  # run from the parent project directory
    python -m ThermoML_results_browser.app  # fixed port 5000

Route modules (Flask Blueprints):
    helpers/agent_routes.py  — /agents/* routes
    helpers/search_routes.py — /search route + smart query helpers

Pure helper modules:
    helpers/thermoml_parsers.py — ThermoML JSON → display-ready dicts
"""

import csv
import importlib
import io
import os
import re
import sys
import threading

from flask import (
    Flask, Response, abort, g, jsonify, redirect, render_template, request,
    send_file, url_for,
)

# Support both package import and direct execution of this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import ThermoML_results_browser

app = Flask(__name__)

_DATABASES_PREPARED = False
_DATABASE_PREPARATION_LOCK = threading.Lock()
_EVAL_STATS_PREPARED = False
_EVAL_STATS_PREPARATION_LOCK = threading.Lock()


def _prepare_databases():
    """Prepare before requests, without making helper/package imports do I/O."""
    global _DATABASES_PREPARED
    if _DATABASES_PREPARED:
        return
    with _DATABASE_PREPARATION_LOCK:
        if not _DATABASES_PREPARED:
            from database_archives import ensure_databases
            ensure_databases(_REPO_ROOT, progress=lambda message: print(message, flush=True))
            _DATABASES_PREPARED = True


def _prepare_eval_stats():
    """Restore the published statistics payload once before serving requests."""
    global _EVAL_STATS_PREPARED
    if _EVAL_STATS_PREPARED:
        return
    with _EVAL_STATS_PREPARATION_LOCK:
        if not _EVAL_STATS_PREPARED:
            from eval_stats_archives import ensure_eval_stats
            ensure_eval_stats(_REPO_ROOT, progress=lambda message: print(message, flush=True))
            _EVAL_STATS_PREPARED = True


@app.before_request
def _prepare_request_databases():
    _prepare_databases()
    _prepare_eval_stats()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CARD_DB_DIR = os.path.join(_REPO_ROOT, 'ThermoML_research_agent', 'card_databases_storage')
DB_PATH = os.path.join(_CARD_DB_DIR, 'ThermoML_index.db')
PER_PAGE = 25

# -- Import parser helpers --
from ThermoML_results_browser.helpers.raw_corpus import read_json as _read_corpus_json
from ThermoML_results_browser.helpers.raw_corpus import read_text as _read_corpus_text
from ThermoML_results_browser.helpers.sqlite_readonly import connect_readonly
from ThermoML_results_browser.helpers.thermoml_parsers import parse_paper_data
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    require_block_id,
    require_block_local_id,
)

def load_paper_data(file_path):
    """Parse a paper directly from the single compressed raw corpus."""
    return parse_paper_data(_read_corpus_json(file_path))


def _extract_subsystem_browser_target(doi, block_number, BLKsubsys_id):
    """Return browser columns/rows for one exact raw subsystem target."""
    extractor = importlib.import_module(
        "card_db_search_tools.basic_search_tools.11_block_data_extractor"
    )
    result = extractor.extract_block_csv(
        doi, block_number, BLKsubsys_id=BLKsubsys_id
    )
    if result["error"] is not None:
        raise LookupError(result["error"])
    metadata = result["metadata"]
    definitions = {
        item["column_name"]: {
            "name": item["column_name"],
            "type": kind,
            "local_id": item[local_field],
        }
        for section, kind, local_field in (
            ("variables", "variable", "BLKvar_id"),
            ("constraints", "constraint", "BLKconstr_id"),
            ("properties", "property", "BLKprop_id"),
        )
        for item in metadata[section]
    }
    columns = []
    for name in result["columns"]:
        columns.append(
            {"name": name, "type": "identity", "local_id": "BLKpoint_id"}
            if name == "BLKpoint_id" else definitions[name]
        )
    dict_rows = list(csv.DictReader(io.StringIO(result["csv_text"])))
    rows = [[row[column["name"]] for column in columns] for row in dict_rows]
    if len(rows) != result["n_rows"]:
        raise ValueError("browser subsystem row count disagrees with extraction")
    return columns, rows, metadata


def _resolve_dynamic_name(name: str, component_name: str | None) -> str:
    """Append compound name to a dynamic entity name."""
    if component_name:
        return f"{name} ({component_name})"
    return name

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    if 'db' not in g:
        g.db = connect_readonly(DB_PATH)
    return g.db


# Give blueprint routes this application's database accessor explicitly.
app.extensions['thermoml_get_db'] = get_db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

from ThermoML_results_browser.helpers.agent_routes import agent_bp
from ThermoML_results_browser.helpers.search_routes import search_bp

app.register_blueprint(agent_bp)
app.register_blueprint(search_bp)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    db = get_db()
    stats = {}
    stats['papers'] = db.execute("SELECT COUNT(*) FROM ref_index").fetchone()[0]
    stats['compounds'] = db.execute(
        "SELECT COUNT(*) FROM compound_registry"
    ).fetchone()[0]
    stats['measurements'] = db.execute("SELECT COUNT(*) FROM block_index").fetchone()[0]
    stats['datapoints'] = db.execute(
        "SELECT COALESCE(SUM(n_datapoints),0) FROM block_index"
    ).fetchone()[0]

    journals = db.execute(
        "SELECT journal, COUNT(*) as cnt FROM ref_index GROUP BY journal ORDER BY cnt DESC"
    ).fetchall()

    recent = db.execute(
        "SELECT doi, title, journal, year FROM ref_index ORDER BY year DESC, doi DESC LIMIT 10"
    ).fetchall()

    # property overview from prop_registry
    top_props = db.execute(
        "SELECT prop_name, n_blocks FROM prop_registry ORDER BY n_blocks DESC LIMIT 15"
    ).fetchall()
    top_props = [(r['prop_name'], r['n_blocks']) for r in top_props]

    return render_template('index.html', stats=stats, journals=journals,
                           recent=recent, top_props=top_props)


@app.route('/papers')
def papers_list():
    db = get_db()
    page = request.args.get('page', 1, type=int)
    journal = request.args.get('journal', '')
    year = request.args.get('year', '')
    prop = request.args.get('prop', '')
    q = request.args.get('q', '').strip()

    query = "SELECT DISTINCT r.doi, r.title, r.first_author, r.journal, r.year, r.n_compounds, r.n_blocks FROM ref_index r"
    joins = []
    wheres = []
    params = []

    if q:
        joins.append("LEFT JOIN block_compounds bc ON bc.doi = r.doi")
        joins.append(
            "LEFT JOIN compound_registry cr ON cr.comp_num_id = bc.comp_num_id"
        )
        wheres.append(
            "(bc.comp_name LIKE ? OR r.title LIKE ? OR r.doi LIKE ? "
            "OR cr.formula LIKE ?)"
        )
        like = f'%{q}%'
        params.extend([like] * 4)

    if prop:
        joins.append("JOIN block_properties bp ON bp.doi = r.doi")
        wheres.append("bp.prop_name LIKE ?")
        params.append(f'%{prop}%')

    if journal:
        wheres.append("r.journal = ?")
        params.append(journal)
    if year:
        wheres.append("r.year = ?")
        params.append(year)

    full = query + " " + " ".join(joins)
    if wheres:
        full += " WHERE " + " AND ".join(wheres)

    count_q = full.replace(
        "SELECT DISTINCT r.doi, r.title, r.first_author, r.journal, r.year, r.n_compounds, r.n_blocks",
        "SELECT COUNT(DISTINCT r.doi)",
    )
    total = db.execute(count_q, params).fetchone()[0]

    full += " ORDER BY r.year DESC, r.doi DESC LIMIT ? OFFSET ?"
    params.extend([PER_PAGE, (page - 1) * PER_PAGE])
    rows = db.execute(full, params).fetchall()

    paper_list = []
    for r in rows:
        # Get compound names for this paper
        comp_rows = db.execute(
            "SELECT DISTINCT bc.comp_name FROM block_compounds bc WHERE bc.doi = ? LIMIT 6",
            (r['doi'],)
        ).fetchall()
        cnames = [c['comp_name'] for c in comp_rows if c['comp_name']]

        paper_list.append({
            'doi': r['doi'], 'title': r['title'],
            'first_author': r['first_author'],
            'journal': r['journal'], 'year': r['year'],
            'compound_names': cnames,
            'n_compounds': r['n_compounds'] or 0,
            'n_measurements': r['n_blocks'] or 0,
        })

    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    journals_opt = db.execute("SELECT DISTINCT journal FROM ref_index ORDER BY journal").fetchall()
    years_opt = db.execute("SELECT DISTINCT year FROM ref_index ORDER BY year DESC").fetchall()

    return render_template('papers.html', papers=paper_list, page=page,
                           total_pages=total_pages, total=total,
                           journal=journal, year=year, q=q, prop=prop,
                           journals=journals_opt, years=years_opt)


@app.route('/paper/<path:doi>')
def paper_detail(doi):
    db = get_db()
    p = db.execute("SELECT * FROM ref_index WHERE doi = ?", (doi,)).fetchone()
    if not p:
        abort(404)

    info = {
        'doi': p['doi'], 'title': p['title'],
        'first_author': p['first_author'],
        'journal': p['journal'], 'year': p['year'],
        'n_compounds': p['n_compounds'],
        'n_blocks': p['n_blocks'],
        'total_datapoints': p['total_datapoints'],
    }

    try:
        data = load_paper_data(p['file_path'])
    except Exception as e:
        data = {'compounds': [], 'measurements': [], 'total_datapoints': 0, 'error': str(e)}

    return render_template('paper.html', paper=info, data=data)


@app.route('/raw-json/<path:doi>')
def raw_json_source(doi):
    """Return an archived source JSON document."""
    db = get_db()
    paper = db.execute(
        "SELECT file_path FROM ref_index WHERE doi = ?", (doi,)
    ).fetchone()
    if not paper:
        abort(404)
    try:
        raw_json = _read_corpus_text(paper['file_path'])
    except (FileNotFoundError, OSError):
        abort(404)
    response = Response(raw_json, content_type='application/json; charset=utf-8')
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/lit/<lit_num_id>')
def lit_redirect(lit_num_id):
    """Resolve GLOBlit_N to its paper page (optionally deep-linking a block)."""
    if not re.fullmatch(r'GLOBlit_[1-9]\d*', lit_num_id):
        abort(404)
    row = get_db().execute(
        "SELECT doi FROM ref_index WHERE lit_num_id = ?", (lit_num_id,)
    ).fetchone()
    if not row:
        abort(404)
    block = request.args.get('block', '')
    anchor = f"#block-{block}" if re.fullmatch(r'(?:PROP|RXN)block_[1-9]\d*', block) else ''
    return redirect(url_for('paper_detail', doi=row['doi']) + anchor)


@app.route('/comp/<comp_num_id>')
def comp_redirect(comp_num_id):
    """Resolve GLOBcomp_N to its compound page."""
    if not re.fullmatch(r'GLOBcomp_[1-9]\d*', comp_num_id):
        abort(404)
    row = get_db().execute(
        "SELECT inchi_key FROM compound_registry WHERE comp_num_id = ?",
        (comp_num_id,),
    ).fetchone()
    if not row or not row['inchi_key']:
        abort(404)
    return redirect(url_for('compound_page', inchikey=row['inchi_key']))


@app.route('/compound/<inchikey>')
def compound_page(inchikey):
    db = get_db()
    # Look up in compound_registry
    cr = db.execute(
        "SELECT comp_num_id, common_name, formula, standard_inchi, "
        "smiles, n_papers FROM compound_registry WHERE inchi_key = ?",
        (inchikey,),
    ).fetchone()
    if not cr:
        abort(404)

    comp = {
        'inchikey': inchikey,
        'names': [cr['common_name']],
        'formula': cr['formula'],
        'inchi': cr['standard_inchi'] or '',
        'smiles': cr['smiles'] or '',
        'n_papers': cr['n_papers'] or 0,
    }

    # Find papers that reference this compound via block_compounds
    rows = db.execute(
        "SELECT DISTINCT r.doi, r.title, r.journal, r.year "
        "FROM block_compounds bc "
        "JOIN ref_index r ON r.doi = bc.doi "
        "WHERE bc.comp_num_id = ? "
        "ORDER BY r.year DESC",
        (cr['comp_num_id'],),
    ).fetchall()

    plist = [{'doi': r['doi'], 'title': r['title'],
              'journal': r['journal'], 'year': r['year']} for r in rows]
    return render_template('compound.html', compound=comp, papers=plist)


# ---------------------------------------------------------------------------
# Analysis Tools
# ---------------------------------------------------------------------------

@app.route('/analysis')
def analysis_tools_page():
    from ThermoML_results_browser.adv_search_calc.analysis_dispatcher import get_tool_definitions
    return render_template('analysis.html', tools=get_tool_definitions())


@app.route('/analysis/tool_definitions')
def analysis_tool_definitions():
    """Return tool definitions for dynamic settings panel rendering."""
    from ThermoML_results_browser.adv_search_calc.analysis_dispatcher import get_tool_definitions
    return jsonify(tools=get_tool_definitions())


@app.route('/analysis/search_papers')
def analysis_search_papers():
    """Quick paper search for the analysis page."""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify(error="Empty query")
    db = get_db()
    like = f'%{q}%'
    rows = db.execute(
        "SELECT doi, title, first_author, journal, year, n_blocks "
        "FROM ref_index "
        "WHERE doi LIKE ? OR title LIKE ? OR first_author LIKE ? "
        "ORDER BY year DESC LIMIT 20",
        (like, like, like),
    ).fetchall()
    # Also search compound names
    if not rows:
        comp_rows = db.execute(
            "SELECT DISTINCT r.doi, r.title, r.first_author, r.journal, r.year, r.n_blocks "
            "FROM block_compounds bc JOIN ref_index r ON r.doi = bc.doi "
            "WHERE bc.comp_name LIKE ? "
            "ORDER BY r.year DESC LIMIT 20",
            (like,),
        ).fetchall()
        rows = comp_rows
    papers = [{'doi': r['doi'], 'title': r['title'], 'first_author': r['first_author'],
               'journal': r['journal'], 'year': r['year'], 'n_blocks': r['n_blocks']}
              for r in rows]
    return jsonify(papers=papers)


@app.route('/analysis/paper_blocks/<path:doi>')
def analysis_paper_blocks(doi):
    """Return declared blocks and search-eligible subsystem targets."""
    from ThermoML_card_json_to_md_compactors._db_access import fetch_card

    db = get_db()
    paper = db.execute(
        "SELECT * FROM ref_index WHERE doi = ?", (doi,)
    ).fetchone()
    if not paper:
        return jsonify(error="Paper not found"), 404
    try:
        data = load_paper_data(paper['file_path'])
        pcs_card = fetch_card("PCS_INDIV", "doi", doi)
    except Exception as exc:
        return jsonify(error=str(exc)), 500

    manifests = pcs_card["blocks_summary"]["derived_indexes"][
        "composition_subsystems"
    ]
    blocks = []
    for measurement in data['measurements']:
        block_number = require_block_id(measurement['block_number'])
        properties_by_id = {
            item['BLKprop_id']: item['name']
            for item in measurement.get('properties', [])
        }
        blocks.append({
            'block_number': block_number,
            'BLKsubsys_id': None,
            'type_label': (
                'Block'
                if measurement['type'] == 'PureOrMixtureData'
                else 'Reaction'
            ),
            'properties': list(properties_by_id.values()),
            'n_datapoints': measurement.get('n_datapoints', 0),
            'components': ', '.join(
                item.get('name', '')
                for item in measurement.get('components', [])
            ),
        })
        for subsystem in manifests[block_number]:
            if subsystem['search_eligible'] is not True:
                continue
            supported = [
                item['BLKprop_id']
                for item in subsystem['property_support']
                if item['role'] == 'bulk_property'
                and item['phase_compatible'] is True
            ]
            blocks.append({
                'block_number': block_number,
                'BLKsubsys_id': require_block_local_id(
                    'subsys', subsystem['BLKsubsys_id']
                ),
                'type_label': (
                    f"{subsystem['effective_system_type'].title()} subsystem"
                ),
                'properties': [
                    properties_by_id.get(local_id, local_id)
                    for local_id in supported
                ],
                'n_datapoints': subsystem['n_points'],
                'components': ', '.join(
                    item['name'] for item in subsystem['retained_components']
                ),
            })
    return jsonify(blocks=blocks)


@app.route('/analysis/block_data/<path:doi>/<block_number>')
def analysis_block_data(doi, block_number):
    """Return the column definitions and rows for one exact block target."""
    try:
        block_number = require_block_id(block_number)
        requested_subsystem = request.args.get('BLKsubsys_id')
        subsystem_id = (
            require_block_local_id('subsys', requested_subsystem)
            if requested_subsystem is not None else None
        )
    except ValueError as exc:
        return jsonify(
            error="ID_REFINEMENT_REQUIRED",
            field=(
                "BLKsubsys_id"
                if request.args.get('BLKsubsys_id') is not None
                else "block_number"
            ),
            received=(
                request.args.get('BLKsubsys_id')
                if request.args.get('BLKsubsys_id') is not None
                else block_number
            ),
            reason=str(exc),
        ), 400

    if subsystem_id is not None:
        try:
            columns, rows, metadata = _extract_subsystem_browser_target(
                doi, block_number, subsystem_id
            )
        except (LookupError, ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(
            doi=doi,
            block_number=block_number,
            BLKsubsys_id=subsystem_id,
            columns=columns,
            rows=rows,
            meta={
                'type': metadata['block_type'],
                'system_type': metadata['system_type'],
                'declared_system_type': metadata['declared_system_type'],
                'properties': [
                    item['prop_ID'] for item in metadata['properties']
                ],
                'components': [
                    item['name'] for item in metadata['compounds']
                ],
                'n_datapoints': metadata['n_datapoints'],
                'constraints': [
                    f"{item['constr_id']} = {item['value']}"
                    for item in metadata['constraints']
                ],
            },
        )

    db = get_db()
    paper = db.execute(
        "SELECT * FROM ref_index WHERE doi = ?", (doi,)
    ).fetchone()
    if not paper:
        return jsonify(error="Paper not found"), 404
    try:
        data = load_paper_data(paper['file_path'])
    except Exception as exc:
        return jsonify(error=str(exc)), 500

    block = next(
        (
            measurement for measurement in data['measurements']
            if measurement['block_number'] == block_number
        ),
        None,
    )
    if block is None:
        return jsonify(error="Block not found"), 404

    return jsonify(
        doi=doi,
        block_number=block_number,
        BLKsubsys_id=None,
        columns=[{'name': item['name'], 'type': item['type']}
                 for item in block['columns']],
        rows=block['rows'],
        meta={
            'type': block['type'],
            'properties': [
                item['name'] for item in block.get('properties', [])
            ],
            'components': [
                item.get('name', '') for item in block.get('components', [])
            ],
            'n_datapoints': block.get('n_datapoints', 0),
            'constraints': [
                item.get('label', '') + ' = ' + str(item.get('value', ''))
                for item in block.get('constraints', [])
            ],
        },
    )


@app.route('/analysis/run', methods=['POST'])
def analysis_run():
    """Run an analysis tool on a data block or custom CSV."""
    from ThermoML_results_browser.adv_search_calc.analysis_dispatcher import dispatch

    payload = request.get_json(force=True)
    tool = payload.get('tool', 'nonideality')
    mode = payload.get('mode', '')
    settings = payload.get('settings', {})

    # --- Custom CSV mode (nonideality only) ---
    if mode == 'custom':
        csv_text = payload.get('csv_text', '')
        if not csv_text.strip():
            return jsonify(success=False, error="No data provided")
        return jsonify(dispatch('nonideality', csv_text=csv_text,
                                columns=[], rows=[]))

    # --- Database block mode ---
    if mode == 'block':
        missing = {'doi', 'block_number'} - payload.keys()
        if missing:
            return jsonify(
                success=False,
                error_code="TOOL_ARGUMENT_REFINEMENT_REQUIRED",
                error=f"Missing required fields: {sorted(missing)}",
            ), 400
        doi = payload['doi']
        block_number = payload['block_number']
        if not isinstance(doi, str) or not doi.strip():
            return jsonify(
                success=False,
                error_code="TOOL_ARGUMENT_REFINEMENT_REQUIRED",
                error="doi must be a non-empty string",
            ), 400
        try:
            block_number = require_block_id(block_number)
        except ValueError as exc:
            return jsonify(
                success=False,
                error_code="ID_REFINEMENT_REQUIRED",
                error=str(exc),
                refinement={
                    "field": "block_number",
                    "received": block_number,
                    "expected": "PROPblock_<n> or RXNblock_<n>",
                    "reason": "Block identifiers are typed and DOI-local.",
                },
            ), 400

        requested_subsystem = payload.get('BLKsubsys_id')
        try:
            subsystem_id = (
                require_block_local_id('subsys', requested_subsystem)
                if requested_subsystem is not None else None
            )
        except ValueError as exc:
            return jsonify(
                success=False,
                error_code="ID_REFINEMENT_REQUIRED",
                error=str(exc),
                refinement={
                    "field": "BLKsubsys_id",
                    "received": requested_subsystem,
                    "expected": "BLKsubsys_<n> or null",
                    "reason": "Subsystem identifiers are block-local.",
                },
            ), 400

        if tool == 'nonideality':
            if subsystem_id is not None:
                try:
                    columns, rows, _ = _extract_subsystem_browser_target(
                        doi, block_number, subsystem_id
                    )
                except (LookupError, ValueError, TypeError) as exc:
                    return jsonify(success=False, error=str(exc)), 404
                return jsonify(dispatch(
                    'nonideality', columns=columns, rows=rows
                ))

            db = get_db()
            paper = db.execute(
                "SELECT * FROM ref_index WHERE doi = ?", (doi,)
            ).fetchone()
            if not paper:
                return jsonify(success=False, error="Paper not found")
            try:
                data = load_paper_data(paper['file_path'])
            except Exception as exc:
                return jsonify(success=False, error=str(exc))

            block = next(
                (
                    measurement for measurement in data['measurements']
                    if measurement['block_number'] == block_number
                ),
                None,
            )
            if block is None:
                return jsonify(success=False, error="Block not found")
            return jsonify(dispatch(
                'nonideality', columns=block['columns'], rows=block['rows']
            ))

        # Agent-based tools receive the same exact target triple as query tools.
        return jsonify(dispatch(
            tool, doi=doi, block_number=block_number,
            BLKsubsys_id=subsystem_id, **settings
        ))

    return jsonify(success=False, error="Unknown mode: " + str(mode))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from database_archives import DatabaseArchiveError
    try:
        _prepare_databases()
    except (DatabaseArchiveError, OSError) as exc:
        raise SystemExit(f"Database preparation failed: {exc}") from exc
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Required index database not found: {DB_PATH}. Restore the supplied database at this path.")
    else:
        print(f"Starting ThermoML Browser on http://localhost:5000")
        app.run(debug=False, use_reloader=False,
                host='127.0.0.1', port=5000, threaded=True)
