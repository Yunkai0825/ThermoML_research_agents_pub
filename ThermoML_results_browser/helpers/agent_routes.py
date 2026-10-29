"""
Agent monitor Flask Blueprint.

All ``/agents/*`` routes — launch, status polling, SSE streaming,
past-run browsing, image/CSV serving, and card lookups.
"""

import csv
import io
import json
import os
import re
import time as _time
from pathlib import Path

from flask import (
    Blueprint, abort, jsonify, render_template, request, Response, send_file,
)

from ThermoML_results_browser import agent_runner
from ThermoML_results_browser.helpers.benchmark_zip import ZipPath, is_within
from ThermoML_results_browser.helpers.answer_eval import load_answer_eval
from ThermoML_results_browser.helpers.sqlite_readonly import connect_readonly
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    GLOBAL_PREFIX_BY_FIELD,
    require_global_id,
    validate_nested_identifiers,
)

agent_bp = Blueprint('agents', __name__)

_AGENT_TYPES = [
    {"key": "main", "label": "Main Agent", "icon": "bi-cpu"},
    {"key": "analysis", "label": "Analysis Agent", "icon": "bi-graph-up"},
    {"key": "query", "label": "Query Agent", "icon": "bi-search"},
]






def _run_scope(value: str | None, default: str = 'user') -> str:
    value = (value or default).strip().lower()
    return value if value in {'user', 'benchmark', 'all'} else default

# Card database directory (one level above this package)
_CARD_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'ThermoML_research_agent', 'card_databases_storage',
)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@agent_bp.route('/agents')
def agents_page():
    busy = agent_runner.is_busy()
    return render_template(
        'agents.html', agent_types=_AGENT_TYPES, busy=busy,
        model_options=agent_runner.AGENT_MODEL_OPTIONS,
    )


@agent_bp.route('/agents/history')
def agents_history():
    """Browse shared freeform runs or read-only benchmark records."""
    scope = _run_scope(request.args.get('scope'))
    if scope == 'all':
        scope = 'user'
    agent_type = (request.args.get('agent_type') or 'analysis').strip().lower()
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        agent_type = 'analysis'
    try:
        page = max(1, int(request.args.get('page') or 1))
    except (TypeError, ValueError):
        page = 1
    page_size = 40
    benchmark_rows = None
    benchmark_stats = None
    if scope == 'benchmark' and agent_type in {'main', 'analysis', 'query'}:
        # Group canonical prompt definitions and their trace/run-backed answers
        # by question. Answer-collection folders are registry inputs, not
        # opaque campaign runs in the sidebar.
        benchmark_rows, benchmark_stats = (
            agent_runner.list_benchmark_questions(agent_type)
        )
        # Standalone campaign sessions still list as plain runs next to the
        # question table; test_run_* answer collections are filtered out.
        runs = agent_runner.list_agent_runs(
            agent_type,
            limit=page_size + 1,
            offset=(page - 1) * page_size,
            scope=scope,
        )
        has_next = len(runs) > page_size
        runs = runs[:page_size]
    else:
        runs = agent_runner.list_agent_runs(
            agent_type,
            limit=page_size + 1,
            offset=(page - 1) * page_size,
            scope=scope,
        )
        has_next = len(runs) > page_size
        runs = runs[:page_size]
    return render_template(
        'agent_history.html',
        agent_types=_AGENT_TYPES,
        selected_agent=agent_type,
        selected_scope=scope,
        runs=runs,
        benchmark_rows=benchmark_rows,
        benchmark_stats=benchmark_stats,
        page=page,
        has_next=has_next,
    )


# ---------------------------------------------------------------------------
# Launch / status
# ---------------------------------------------------------------------------

@agent_bp.route('/agents/launch', methods=['POST'])
def agents_launch():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify(error="Question is required."), 400
    if agent_runner.is_busy():
        return jsonify(error="An agent run is already in progress."), 409
    agent_type = data.get('agent_type', 'analysis')
    if agent_type not in ('analysis', 'query', 'main'):
        return jsonify(error="Invalid agent type."), 400
    try:
        model_override = agent_runner.normalize_model_choice(data.get('model'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    run_verdict = data.get('run_verdict', True)
    time_overrides = data.get('time_overrides') or {}
    # Whitelist allowed override keys
    allowed_keys = {'MAX_TOOL_ITERATIONS', 'MAX_TURN_SECONDS',
                    'L1_MAX_ITERATIONS', 'L1_MAX_SECONDS'}
    time_overrides = {k: v for k, v in time_overrides.items()
                      if k in allowed_keys and isinstance(v, (int, float))}
    try:
        run_id = agent_runner.start_run(
            question, agent_type=agent_type,
            run_verdict=run_verdict, time_overrides=time_overrides,
            model_override=model_override,
        )
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(run_id=run_id, model=model_override or '')


@agent_bp.route('/agents/status/<run_id>')
def agents_status(run_id):
    run_obj = agent_runner.get_run(run_id)
    if not run_obj:
        return jsonify(error="Run not found."), 404

    wm_changed = run_obj.check_working_memory()
    wm_content = run_obj.get_working_memory() if wm_changed else None

    # Poll history.md for live tool-call activity (only while running)
    if run_obj.status in ('starting', 'running'):
        run_obj.check_history_for_activity()

    # Determine the max time for progress bar scaling
    _agent_max_times = {'analysis': 1200, 'query': 1000, 'main': 2400}
    max_turn = run_obj.time_overrides.get(
        'MAX_TURN_SECONDS',
        _agent_max_times.get(run_obj.agent_type, 1200),
    )

    images, data_files = run_obj.artifacts()
    payload = {
        'status': run_obj.status,
        'elapsed': round(run_obj.elapsed, 1),
        'max_turn_seconds': max_turn,
        'images': images,
        'data_files': data_files,
        'session_dir': run_obj.session_dir,
        'model': run_obj.model_override or '',
        'progress_log': run_obj.progress_log[-5:],
    }

    payload['reasoning_snapshots'] = run_obj.reasoning_snapshots[-10:]
    payload['reasoning_md'] = run_obj.get_reasoning_md()

    if wm_content is not None:
        payload['working_memory'] = wm_content

    if run_obj.status == 'completed' and run_obj.result:
        payload['result_md'] = run_obj.get_result_md()
        payload['history_md'] = run_obj.get_history_md()
        payload['memory_md'] = run_obj.get_working_memory()
        payload['stats_md'] = run_obj.get_reference_stats_md()
        payload['iterations'] = getattr(run_obj.result, 'iterations', None)
        payload['total_seconds'] = round(
            getattr(run_obj.result, 'elapsed_seconds', run_obj.elapsed), 1)
        payload['timed_out'] = getattr(run_obj.result, 'timed_out', False)
        payload['dir_name'] = (
            Path(run_obj.session_dir).name if run_obj.session_dir else '')
    elif run_obj.status == 'error':
        payload['error'] = run_obj.error

    return jsonify(payload)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

_STRIP_XML_RE = re.compile(r'</?[a-zA-Z_][a-zA-Z0-9_]*[^>]*/?>',)

def _sanitize_line(line: str) -> str:
    """Strip XML-style tags and collapse whitespace for safe display."""
    out = _STRIP_XML_RE.sub('', line)
    return out.strip()


def _sse(data: dict, event_id: int | None = None) -> str:
    """Format a dict as an SSE data line."""
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}data: {json.dumps(data, default=str)}\n\n"


@agent_bp.route('/agents/stream/<run_id>')
def agents_stream(run_id):
    """Server-Sent Events endpoint for real-time agent progress."""
    run_obj = agent_runner.get_run(run_id)
    if not run_obj:
        return jsonify(error="Run not found."), 404

    try:
        resume_log_seq = int(request.headers.get('Last-Event-ID') or 0)
    except (TypeError, ValueError):
        resume_log_seq = 0

    def generate():
        _INTERVAL = 1.5          # seconds between checks
        _MAX_IDLE = 30 * 60      # give up after 30 min of no change
        idle_since = _time.time()
        _prev_stats_md = ''       # track changes to avoid resending identical data
        _prev_reasoning_md = ''
        _last_log_seq = resume_log_seq

        # Determine the effective max for this agent_type
        _agent_max = {'analysis': 1200, 'query': 1000, 'main': 2400}
        max_turn = _agent_max.get(run_obj.agent_type, 1200)
        if run_obj.time_overrides:
            max_turn = run_obj.time_overrides.get('MAX_TURN_SECONDS', max_turn)

        while True:
            ro = agent_runner.get_run(run_id)
            if not ro:
                yield _sse({'type': 'error', 'msg': 'Run not found'})
                break

            events_sent = False

            # ── new history lines (tool-call progress) ──────────────
            new_hist = ro.get_new_history_lines()
            if new_hist:
                for ln in new_hist:
                    s = ln.strip()
                    if s.startswith('### Step ') or s.startswith('- **Result:**'):
                        yield _sse({
                            'type': 'history',
                            'text': _sanitize_line(s),
                        })
                        events_sent = True

            # ── new reasoning tokens (LLM thinking) ─────────────────
            new_reason = ro.get_new_reasoning_text()
            if new_reason:
                cleaned = _sanitize_line(new_reason)
                if len(cleaned) > 300:
                    cleaned = '…' + cleaned[-300:]
                if cleaned:
                    yield _sse({
                        'type': 'reasoning',
                        'text': cleaned,
                    })
                    events_sent = True

            # ── live reference stats (API call summary) ─────────────
            stats_md = ro.get_reference_stats_md()
            if stats_md and stats_md != _prev_stats_md:
                _prev_stats_md = stats_md
                yield _sse({
                    'type': 'ref_stats',
                    'md': stats_md,
                })
                events_sent = True

            # ── live reasoning full MD ──────────────────────────────
            reasoning_md = ro.get_reasoning_md()
            if reasoning_md and reasoning_md != _prev_reasoning_md:
                _prev_reasoning_md = reasoning_md
                yield _sse({
                    'type': 'reasoning_full',
                    'md': reasoning_md,
                })
                events_sent = True

            # ── terminal log lines ──────────────────────────────────
            new_seq, new_logs = ro.logs_since(_last_log_seq)
            if new_logs:
                _last_log_seq = new_seq
                idle_since = _time.time()
                filtered_logs = new_logs
                if filtered_logs:
                    yield _sse({
                        'type': 'terminal_log',
                        'lines': filtered_logs,
                    }, event_id=new_seq)
                    events_sent = True

            # ── status heartbeat (always) ───────────────────────────
            ro.check_history_for_activity()
            images, data_files = ro.artifacts()
            yield _sse({
                'type': 'status',
                'status': ro.status,
                'elapsed': round(ro.elapsed, 1),
                'n_images': len(images),
                'n_data': len(data_files),
                'max_turn_seconds': max_turn,
                'model': ro.model_override or '',
            })

            # ── terminal states ─────────────────────────────────────
            if ro.status == 'completed':
                payload = {
                    'type': 'done',
                    'status': 'completed',
                    'iterations': getattr(ro.result, 'iterations', None),
                    'total_seconds': round(
                        getattr(ro.result, 'elapsed_seconds', ro.elapsed), 1),
                    'timed_out': getattr(ro.result, 'timed_out', False),
                    'result_md': ro.get_result_md(),
                    'images': images,
                    'data_files': data_files,
                    'dir_name': Path(ro.session_dir).name if ro.session_dir else '',
                    'agent_type': ro.agent_type,
                    'model': ro.model_override or '',
                }
                payload['history_md'] = ro.get_history_md()
                payload['memory_md'] = ro.get_working_memory()
                payload['stats_md'] = ro.get_reference_stats_md()
                if ro.session_dir:
                    payload['workflow'] = agent_runner.list_workflow_figures(
                        Path(ro.session_dir))
                yield _sse(payload)
                break
            elif ro.status == 'error':
                yield _sse({
                    'type': 'done',
                    'status': 'error',
                    'error': ro.error or 'Unknown error',
                    'model': ro.model_override or '',
                })
                break

            # Idle timeout guard
            if events_sent:
                idle_since = _time.time()
            elif (_time.time() - idle_since) > _MAX_IDLE:
                yield _sse({'type': 'error', 'msg': 'Idle timeout'})
                break

            _time.sleep(_INTERVAL)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ---------------------------------------------------------------------------
# Image / file serving (live runs)
# ---------------------------------------------------------------------------

@agent_bp.route('/agents/image/<run_id>/<path:filename>')
def agents_image(run_id, filename):
    """Serve an image from a running or completed agent session."""
    run_obj = agent_runner.get_run(run_id)
    if not run_obj or not run_obj.session_dir:
        abort(404)
    img_path = agent_runner.resolve_artifact_file(
        run_obj.session_dir, filename, 'plots')
    if img_path is None:
        abort(404)
    return send_file(str(img_path))


@agent_bp.route('/agents/live_csv/<run_id>/<path:filename>')
def agents_live_csv(run_id, filename):
    """Return one CSV generated by a just-completed in-memory run."""
    run_obj = agent_runner.get_run(run_id)
    if not run_obj or not run_obj.session_dir:
        abort(404)
    csv_path = agent_runner.resolve_artifact_file(
        run_obj.session_dir, filename, 'data')
    if csv_path is None:
        abort(404)
    try:
        with csv_path.open('r', encoding='utf-8', errors='replace') as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return jsonify(columns=[], rows=[])
        return jsonify(columns=rows[0], rows=rows[1:])
    except Exception as exc:
        return jsonify(error=str(exc)), 500


# ---------------------------------------------------------------------------
# Multi-agent browsing API routes
# ---------------------------------------------------------------------------

@agent_bp.route('/agents/runs/<agent_type>')
def agents_runs_list(agent_type):
    """Return JSON list of past runs for an agent type."""
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        return jsonify(error="Unknown agent type"), 400
    scope = _run_scope(request.args.get('scope'), default='all')
    runs = agent_runner.list_agent_runs(agent_type, limit=100, scope=scope)
    return jsonify(runs=runs)


@agent_bp.route('/agents/detail/<agent_type>/<dir_name>')
def agents_run_detail(agent_type, dir_name):
    """Return JSON detail (result_md, memory_md, images, csvs) for a run."""
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        return jsonify(error="Unknown agent type"), 400
    sub_run = request.args.get('sub_run')
    scope = _run_scope(request.args.get('scope'), default='all')
    data = agent_runner.get_run_files(
        agent_type, dir_name, sub_run=sub_run, scope=scope,
        include_full_tool_history=True)
    if "error" in data:
        return jsonify(data), 404
    return jsonify(data)


@agent_bp.route('/agents/eval/<agent_type>/<dir_name>')
def agents_run_eval(agent_type, dir_name):
    """Return the saved evaluation in a run's eval/ folder for the Eval switch.

    The eval files place their highlights on the answer folder's own
    result.md, so an answer shown from another file (a flat Q*_result.md or a
    newer result_R<n>.md) has none.
    """
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        return jsonify(error="Unknown agent type"), 400
    scope = _run_scope(request.args.get('scope'), default='all')
    sub_run = request.args.get('sub_run') or ''
    if (dir_name in ('.', '..') or sub_run in ('.', '..')
            or '/' in sub_run or '\\' in sub_run
            or ':' in sub_run or '\x00' in sub_run):
        abort(404)
    answer_dir = agent_runner.resolve_run_dir(agent_type, dir_name, scope=scope)
    if answer_dir is None:
        abort(404)
    if sub_run:
        nested = answer_dir / sub_run
        if not is_within(nested, answer_dir):
            abort(404)
        answer_dir = nested
    if (not (answer_dir / 'result.md').is_file()
            or agent_runner._latest_result_md_path(answer_dir).name != 'result.md'):
        return jsonify(numbers=None, claims=None, lines={})
    return jsonify(load_answer_eval(answer_dir))


@agent_bp.route('/agents/delete/<agent_type>/<dir_name>', methods=['POST'])
def agents_run_delete(agent_type, dir_name):
    """Delete one shared freeform run; benchmark data is unreachable."""
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        return jsonify(error="Unknown agent type"), 400
    if not agent_runner.delete_user_run(agent_type, dir_name):
        return jsonify(error="Freeform run not found or could not be removed."), 404
    return jsonify(ok=True)


@agent_bp.route('/agents/csv/<agent_type>/<dir_name>/<path:filename>')
def agents_csv_data(agent_type, dir_name, filename):
    """Parse a CSV file from a run's data/ folder and return as JSON."""
    scope = _run_scope(request.args.get('scope'), default='all')
    sub_run = os.path.basename(request.args.get('sub_run') or '')
    base = agent_runner.resolve_run_artifact_dir(
        agent_type, dir_name, sub_run=sub_run, scope=scope)
    if base is None:
        abort(404)
    csv_path = agent_runner.resolve_artifact_file(base, filename, "data")
    if csv_path is None:
        abort(404)
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return jsonify(columns=[], rows=[])
        return jsonify(columns=rows[0], rows=rows[1:])
    except Exception as exc:
        return jsonify(error=str(exc)), 500


def _send_run_file(path, **kwargs):
    """Serve an archive member from memory, or an ordinary file unchanged."""
    if isinstance(path, ZipPath):
        kwargs.setdefault("download_name", path.name)
        kwargs.setdefault("last_modified", path.stat().st_mtime)
        return send_file(io.BytesIO(path.read_bytes()), **kwargs)
    return send_file(str(path), **kwargs)


@agent_bp.route('/agents/img/<agent_type>/<dir_name>/<path:filename>')
def agents_img_serve(agent_type, dir_name, filename):
    """Serve an image from a run's plots/ folder."""
    scope = _run_scope(request.args.get('scope'), default='all')
    sub_run = os.path.basename(request.args.get('sub_run') or '')
    base = agent_runner.resolve_run_artifact_dir(
        agent_type, dir_name, sub_run=sub_run, scope=scope)
    if base is None:
        abort(404)
    img_path = agent_runner.resolve_artifact_file(base, filename, "plots")
    if img_path is None:
        abort(404)
    return _send_run_file(img_path)


@agent_bp.route('/agents/workflow/<agent_type>/<dir_name>/<path:filename>')
def agents_workflow_serve(agent_type, dir_name, filename):
    """Serve a compaction-workflow artifact (Sankey png/html, edge TSV)
    from a run's workflow/ folder."""
    if agent_type not in agent_runner.AGENT_OUTPUT_DIRS:
        abort(404)
    scope = _run_scope(request.args.get('scope'), default='all')
    sub_run = os.path.basename(request.args.get('sub_run') or '')
    base = agent_runner.resolve_run_artifact_dir(
        agent_type, dir_name, sub_run=sub_run, scope=scope)
    if base is None:
        abort(404)
    safe_file = os.path.basename(filename)
    if os.path.splitext(safe_file)[1].lower() not in (".png", ".html", ".tsv"):
        abort(404)
    wf_path = base / "workflow" / safe_file
    if not wf_path.is_file():
        abort(404)
    if not is_within(wf_path, base):
        abort(403)
    if wf_path.suffix.lower() == ".tsv":
        return _send_run_file(wf_path, mimetype="text/tab-separated-values",
                         as_attachment=True, download_name=safe_file)
    return _send_run_file(wf_path)


# ---------------------------------------------------------------------------
# Card lookup
# ---------------------------------------------------------------------------

@agent_bp.route('/agents/card/<entity_type>/<global_id>')
def agents_card_lookup(entity_type, global_id):
    """Return ID/DK card JSON for a property, measurement, variable, or constraint."""
    iddk_dir = os.path.join(_CARD_DB_DIR, 'Individual_cards_dbs')
    global_field_by_entity = {
        'prop': 'prop_num_id',
        'meas': 'meas_num_id',
        'var': 'var_num_id',
        'constr': 'constr_num_id',
    }
    if entity_type not in global_field_by_entity:
        return jsonify(
            error_code="ID_REFINEMENT_REQUIRED",
            error="entity_type must be one of: prop, meas, var, constr",
            refinement={
                "field": "entity_type",
                "received": entity_type,
                "expected": "prop | meas | var | constr",
                "reason": "unknown canonical ID namespace",
            },
        ), 400
    global_field = global_field_by_entity[entity_type]
    try:
        global_id = require_global_id(global_field, global_id)
    except ValueError as exc:
        return jsonify(
            error_code="ID_REFINEMENT_REQUIRED",
            error=str(exc),
            refinement={
                "field": global_field,
                "received": global_id,
                "expected": GLOBAL_PREFIX_BY_FIELD[global_field] + "<positive integer>",
                "reason": "wrong-scope, unprefixed, or retired identifier",
            },
        ), 400

    if entity_type == 'prop':
        db_path = os.path.join(iddk_dir, 'PCS_ID_DK.db')
        if not os.path.exists(db_path):
            return jsonify(error="Property card DB not found"), 404
        conn = connect_readonly(db_path)
        row = conn.execute(
            "SELECT prop_num_id, prop_id, prop_name, prop_group, json_data "
            "FROM cards WHERE prop_num_id = ?", (global_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify(error=f"Property {global_id} not found"), 404
        if not row['json_data']:
            raise ValueError(f"Property {global_id} has empty card JSON")
        card_json = json.loads(row['json_data'])
        validate_nested_identifiers(card_json, path=f"PCS_ID_DK[{global_id}]")
        return jsonify(
            entity_type='prop', global_id=row['prop_num_id'],
            entity_id=row['prop_id'], name=row['prop_name'],
            group=row['prop_group'], card=card_json,
        )

    elif entity_type == 'meas':
        db_path = os.path.join(iddk_dir, 'MTDKS_ID_DK.db')
        if not os.path.exists(db_path):
            return jsonify(error="Measurement card DB not found"), 404
        conn = connect_readonly(db_path)
        row = conn.execute(
            "SELECT meas_num_id, meas_id, method_name, method_type, json_data "
            "FROM cards WHERE meas_num_id = ?", (global_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify(error=f"Measurement {global_id} not found"), 404
        if not row['json_data']:
            raise ValueError(f"Measurement {global_id} has empty card JSON")
        card_json = json.loads(row['json_data'])
        validate_nested_identifiers(card_json, path=f"MTDKS_ID_DK[{global_id}]")
        return jsonify(
            entity_type='meas', global_id=row['meas_num_id'],
            entity_id=row['meas_id'], name=row['method_name'],
            group=row['method_type'], card=card_json,
        )

    elif entity_type == 'var':
        db_path = os.path.join(_CARD_DB_DIR, 'ThermoML_index.db')
        conn = connect_readonly(db_path)
        row = conn.execute(
            "SELECT var_num_id, var_id, var_name, var_type_key, n_blocks "
            "FROM var_registry WHERE var_num_id = ?", (global_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify(error=f"Variable {global_id} not found"), 404
        return jsonify(
            entity_type='var', global_id=row['var_num_id'],
            entity_id=row['var_id'], name=row['var_name'],
            group=row['var_type_key'], n_blocks=row['n_blocks'],
            card=None,
        )

    elif entity_type == 'constr':
        db_path = os.path.join(_CARD_DB_DIR, 'ThermoML_index.db')
        conn = connect_readonly(db_path)
        row = conn.execute(
            "SELECT constr_num_id, constr_id, constr_name, constr_type_key, n_blocks "
            "FROM constr_registry WHERE constr_num_id = ?", (global_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify(error=f"Constraint {global_id} not found"), 404
        return jsonify(
            entity_type='constr', global_id=row['constr_num_id'],
            entity_id=row['constr_id'], name=row['constr_name'],
            group=row['constr_type_key'], n_blocks=row['n_blocks'],
            card=None,
        )

    raise AssertionError(f"Unhandled validated entity_type: {entity_type}")
