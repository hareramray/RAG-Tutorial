"""Meaningful SQL boundary, exact-arithmetic, and unknown-data checks; no API calls."""
import sqlite3

import pytest

from rag_workshop.structured import (
    QUERIES, _authorize, create_demo_database, execute_plan, safe_select,
    sql_evidence, validate_plan, validate_select, parse_plan_response,
)


@pytest.fixture
def database(tmp_path):
    return create_demo_database(tmp_path / "northstar.sqlite")


def test_account_joins_and_null_price(database):
    result = execute_plan(database, {"query_id": "account_overview", "parameters": {"account_name": "Cedar Labs"}})
    row = result["rows"][0]
    assert (row["purchased_users"], row["active_users"], row["unused_users"]) == (8, 6, 2)
    assert row["annual_list_price_inr_ex_tax"] == 230400
    unknown = execute_plan(database, {"query_id": "account_overview", "parameters": {"account_name": "Orion Systems"}})
    assert unknown["rows"][0]["annual_list_price_inr_ex_tax"] is None
    assert "NULL means unknown" in sql_evidence(unknown)[0]["text"]


def test_quote_minimum_and_paid_orders(database):
    quote = execute_plan(database, {"query_id": "team_quote", "parameters": {"requested_users": 3}})["rows"][0]
    assert (quote["requested_users"], quote["billable_users"], quote["annual_list_price_inr_ex_tax"]) == (3, 5, 144000)
    paid = execute_plan(database, {"query_id": "paid_orders", "parameters": {"account_name": "Cedar Labs"}})["rows"][0]
    assert (paid["paid_order_count"], paid["paid_order_total_inr_ex_tax"]) == (2, 460800)


@pytest.mark.parametrize("plan", [
    {"query_id": "unknown", "parameters": {}},
    {"query_id": "team_quote", "parameters": {"requested_users": True}},
    {"query_id": "team_quote", "parameters": {"requested_users": 0}},
    {"query_id": "account_overview", "parameters": {"account_name": ""}},
    {"query_id": "region_summary", "parameters": {"sql": "DELETE FROM accounts"}},
    {"query_id": "unsupported", "parameters": {}, "sql": "SELECT 1"},
    {"query_id": "account_overview", "parameters": {}},
])
def test_plan_schema_rejects_extra_invalid_and_missing_fields(plan):
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_plan_parser_handles_one_fenced_object_without_executing_prose():
    payload = '{"query_id":"region_summary","parameters":{}}'
    assert parse_plan_response(payload)["query_id"] == "region_summary"
    assert parse_plan_response("```json\n" + payload + "\n```\nThis selects the regional summary.")["query_id"] == "region_summary"


@pytest.mark.parametrize("text", [
    '```json\n{"query_id":"unsupported","parameters":{}}\n```\n```json\n{}\n```',
    '{"query_id":"unsupported","parameters":{}} {"query_id":"region_summary","parameters":{}}',
    '```json\n{"query_id":"unsupported","parameters":{}}\n```\nAnother plan is {}.',
    'Here is my unstructured guess',
])
def test_plan_parser_rejects_ambiguous_or_non_json_text(text):
    with pytest.raises(ValueError):
        parse_plan_response(text)


@pytest.mark.parametrize("sql", [
    "DELETE FROM accounts",
    "SELECT * FROM accounts; DROP TABLE accounts",
    "SELECT * FROM accounts; SELECT * FROM plans",
    "PRAGMA writable_schema=ON",
    "ATTACH DATABASE 'other.sqlite' AS other",
    "SELECT load_extension('example')",
    "SELECT readfile('private.txt')",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM other.accounts",
    "WITH a AS (SELECT * FROM accounts) SELECT * FROM a",
    "SELECT * FROM (SELECT * FROM accounts)",
    "SELECT * FROM accounts UNION SELECT * FROM accounts",
])
def test_ast_rejects_writes_external_sources_and_broader_grammar(database, sql):
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        safe_select(database, sql)
    assert safe_select(database, "SELECT COUNT(*) AS n FROM accounts")["rows"][0]["n"] == 4


def test_parameters_remain_data_empty_results_abstain(database):
    plan = {"query_id": "account_overview", "parameters": {"account_name": "Cedar Labs' OR 1=1 --"}}
    assert execute_plan(database, plan)["rows"] == []
    assert sql_evidence(execute_plan(database, plan)) == []
    assert sql_evidence(execute_plan(database, {"query_id": "unsupported", "parameters": {}})) == []


def test_authorizer_is_a_separate_boundary(database):
    # Bypass the AST intentionally in this test to verify the SQLite authorizer.
    connection = sqlite3.connect(database)
    connection.set_authorizer(_authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM accounts")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("PRAGMA table_info(accounts)")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT randomblob(1000000)")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT name FROM sqlite_master")
        assert connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 4
    finally:
        connection.close()


def test_row_budget_and_compute_budget(database):
    result = safe_select(database, "SELECT * FROM accounts ORDER BY account_id", max_rows=2)
    assert result["truncated"] and len(result["rows"]) == 2
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        safe_select(database, "SELECT COUNT(*) FROM seats AS a CROSS JOIN seats AS b CROSS JOIN seats AS c",
                    max_vm_steps=100)


def test_query_result_citations_change_with_params(database):
    a = safe_select(database, QUERIES["team_quote"], {"requested_users": 5})
    b = safe_select(database, QUERIES["team_quote"], {"requested_users": 8})
    assert a["id"] != b["id"]
    evidence = sql_evidence(a)[0]
    assert evidence["id"] == a["id"]
    assert "Bound parameters:" in evidence["text"]
    assert "144000" in evidence["text"]


def test_creator_never_overwrites_existing_file(database):
    original = database.read_bytes()
    with pytest.raises(FileExistsError):
        create_demo_database(database)
    assert database.read_bytes() == original
