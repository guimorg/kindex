"""Tests for SQLite store."""

import os
import time
from datetime import datetime, timezone

import pytest

from kindex.config import Config
from kindex.store import Store


@pytest.fixture
def store(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    s = Store(cfg)
    yield s
    s.close()


class TestNodeOperations:
    def test_add_and_get(self, store):
        nid = store.add_node("Test Topic", content="Some content", node_type="concept")
        node = store.get_node(nid)
        assert node["title"] == "Test Topic"
        assert node["content"] == "Some content"
        assert node["type"] == "concept"

    def test_add_with_domains(self, store):
        nid = store.add_node("D", domains=["eng", "research"])
        node = store.get_node(nid)
        assert node["domains"] == ["eng", "research"]

    def test_get_by_title(self, store):
        store.add_node("Unique Title", node_id="ut1")
        node = store.get_node_by_title("Unique Title")
        assert node["id"] == "ut1"
        assert store.get_node_by_title("unique title") is not None  # case insensitive

    def test_update_node(self, store):
        nid = store.add_node("Original")
        store.update_node(nid, title="Updated", weight=0.9)
        node = store.get_node(nid)
        assert node["title"] == "Updated"
        assert node["weight"] == 0.9

    def test_delete_node(self, store):
        nid = store.add_node("Doomed")
        store.delete_node(nid)
        assert store.get_node(nid) is None

    def test_all_nodes(self, store):
        store.add_node("A", node_type="concept")
        store.add_node("B", node_type="skill")
        store.add_node("C", node_type="concept")
        assert len(store.all_nodes()) == 3
        assert len(store.all_nodes(node_type="concept")) == 2

    def test_all_nodes_filters_by_source_event_time(self, store):
        store.add_node("Before", node_id="before", prov_when="2026-06-09T23:59:59Z")
        store.add_node("Start", node_id="start", prov_when="2026-06-10T00:00:00Z")
        store.add_node("End", node_id="end", prov_when="2026-06-24T00:00:00Z")

        nodes = store.all_nodes(
            since="2026-06-10T00:00:00Z",
            until="2026-06-24T00:00:00Z",
            order="event_time_asc",
        )

        assert [node["id"] for node in nodes] == ["start"]

    def test_all_nodes_falls_back_to_created_at(self, store):
        store.add_node("Legacy", node_id="legacy")
        store.conn.execute(
            "UPDATE nodes SET prov_when = '', created_at = ? WHERE id = ?",
            ("2026-06-12T10:00:00", "legacy"),
        )
        store.conn.commit()

        nodes = store.all_nodes(since="2026-06-12", until="2026-06-13")

        assert [node["id"] for node in nodes] == ["legacy"]

    def test_all_nodes_paginates_with_stable_event_order(self, store):
        for index in range(4):
            store.add_node(
                f"Node {index}",
                node_id=f"node-{index}",
                prov_when=f"2026-06-1{index}T00:00:00Z",
            )

        first = store.all_nodes(limit=2, offset=0, order="event_time_asc")
        second = store.all_nodes(limit=2, offset=2, order="event_time_asc")

        assert [node["id"] for node in first] == ["node-0", "node-1"]
        assert [node["id"] for node in second] == ["node-2", "node-3"]
        assert store.count_nodes(since="2026-06-10", until="2026-06-14") == 4

    def test_page_nodes_reads_total_and_nodes_from_one_snapshot(self, store, monkeypatch):
        writer = Store(store.config)
        writer.conn
        original_count = store.count_nodes

        def count_then_write(*args, **kwargs):
            total = original_count(*args, **kwargs)
            writer.add_node("Concurrent", node_id="concurrent")
            return total

        monkeypatch.setattr(store, "count_nodes", count_then_write)
        try:
            nodes, total = store.page_nodes()
        finally:
            writer.close()

        assert nodes == []
        assert total == 0
        assert original_count() == 1

    def test_page_nodes_does_not_commit_caller_transaction(self, store):
        store.conn.execute("BEGIN")

        nodes, total = store.page_nodes()

        assert nodes == []
        assert total == 0
        assert store.conn.in_transaction
        store.conn.rollback()

    def test_all_nodes_preserves_fractional_second_precision(self, store):
        store.add_node("Later", node_id="later", prov_when="2026-06-10T00:00:00.900Z")
        store.add_node("Earlier", node_id="earlier", prov_when="2026-06-10T00:00:00.100Z")

        nodes = store.all_nodes(
            since="2026-06-10T00:00:00.050Z",
            until="2026-06-10T00:00:00.500Z",
            order="event_time_asc",
        )

        assert [node["id"] for node in nodes] == ["earlier"]

    def test_add_node_normalizes_event_time_to_milliseconds(self, store):
        store.add_node(
            "Rounded",
            node_id="rounded",
            prov_when="2026-06-10T03:00:00.123500+03:00",
        )

        assert store.get_node("rounded")["prov_when"] == "2026-06-10T00:00:00.124Z"

    def test_add_node_default_event_time_is_utc_under_non_utc_timezone(self, store):
        original_timezone = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/Sao_Paulo"
            time.tzset()
            before = datetime.now(timezone.utc)
            store.add_node("UTC default", node_id="utc-default")
            after = datetime.now(timezone.utc)
        finally:
            if original_timezone is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_timezone
            time.tzset()

        event_time = datetime.fromisoformat(store.get_node("utc-default")["prov_when"])
        assert before.timestamp() - 0.001 <= event_time.timestamp()
        assert event_time.timestamp() <= after.timestamp() + 0.001

    def test_store_normalizes_sub_millisecond_query_boundaries(self, store):
        store.add_node(
            "Rounded",
            node_id="rounded",
            prov_when="2026-06-10T00:00:00.000500Z",
        )

        nodes = store.all_nodes(
            since="2026-06-10T00:00:00.000500Z",
            until="2026-06-10T00:00:00.001500Z",
        )

        assert [node["id"] for node in nodes] == ["rounded"]

    def test_all_nodes_normalizes_timezone_offsets(self, store):
        store.add_node("Same instant", node_id="same", prov_when="2026-06-10T03:00:00+03:00")
        store.add_node("Later", node_id="later", prov_when="2026-06-10T00:00:00.001Z")

        nodes = store.all_nodes(
            since="2026-06-10T00:00:00Z",
            until="2026-06-10T00:00:00.001Z",
            order="event_time_asc",
        )

        assert [node["id"] for node in nodes] == ["same"]

    def test_temporal_filter_uses_event_time_expression_index(self, store):
        where, params = store._node_filters(since="2026-06-10T00:00:00Z")
        plan = store.conn.execute(
            f"EXPLAIN QUERY PLAN SELECT * FROM nodes WHERE {where}", params
        ).fetchall()

        assert any("idx_nodes_event_time" in row["detail"] for row in plan)

    def test_all_nodes_rejects_invalid_order_and_offset(self, store):
        with pytest.raises(ValueError, match="offset"):
            store.all_nodes(offset=-1)
        with pytest.raises(ValueError, match="order"):
            store.all_nodes(order="newest")
        with pytest.raises(ValueError, match="since must be earlier"):
            store.all_nodes(since="2026-06-11", until="2026-06-10")

    def test_recent_nodes(self, store):
        store.add_node("Old")
        store.add_node("New")
        recent = store.recent_nodes(n=1)
        assert len(recent) == 1
        assert recent[0]["title"] == "New"

    def test_node_ids(self, store):
        store.add_node("A", node_id="a1")
        store.add_node("B", node_id="b2")
        ids = store.node_ids()
        assert "a1" in ids
        assert "b2" in ids


class TestEdgeOperations:
    def test_add_edge_bidirectional(self, store):
        store.add_node("A", node_id="a")
        store.add_node("B", node_id="b")
        store.add_edge("a", "b", provenance="test")
        assert len(store.edges_from("a")) == 1
        assert len(store.edges_to("a")) == 1  # bidirectional creates reverse

    def test_edges_from(self, store):
        store.add_node("X", node_id="x")
        store.add_node("Y", node_id="y")
        store.add_edge("x", "y", edge_type="implements", weight=0.9)
        edges = store.edges_from("x")
        assert edges[0]["to_id"] == "y"
        assert edges[0]["type"] == "implements"
        assert edges[0]["weight"] == 0.9

    def test_orphans(self, store):
        store.add_node("Lonely", node_id="lonely")
        store.add_node("Connected", node_id="conn")
        store.add_node("Also Connected", node_id="also")
        store.add_edge("conn", "also")
        orphans = store.orphans()
        assert len(orphans) == 1
        assert orphans[0]["id"] == "lonely"


class TestFTS:
    def test_fts_search(self, store):
        store.add_node("Stigmergy Coordination", content="Agents communicate indirectly",
                        node_id="stig")
        store.add_node("Database Design", content="Schema normalization", node_id="db")
        results = store.fts_search("stigmergy")
        assert len(results) >= 1
        assert results[0]["id"] == "stig"

    def test_fts_no_results(self, store):
        store.add_node("Something", content="content")
        results = store.fts_search("zzzznonexistent")
        assert results == []


class TestTagFiltering:
    def test_all_nodes_filter_single_tag(self, store):
        store.add_node("A", domains=["python", "web"])
        store.add_node("B", domains=["python", "ml"])
        store.add_node("C", domains=["rust"])
        results = store.all_nodes(tags=["python"])
        assert len(results) == 2

    def test_all_nodes_filter_multiple_tags_and_logic(self, store):
        store.add_node("A", domains=["python", "web"])
        store.add_node("B", domains=["python", "ml"])
        results = store.all_nodes(tags=["python", "web"])
        assert len(results) == 1
        assert results[0]["title"] == "A"

    def test_all_nodes_filter_no_match(self, store):
        store.add_node("A", domains=["python"])
        results = store.all_nodes(tags=["java"])
        assert len(results) == 0

    def test_add_node_with_tags_alias(self, store):
        nid = store.add_node("X", tags=["alpha", "beta"])
        node = store.get_node(nid)
        assert "alpha" in node["domains"]
        assert "beta" in node["domains"]
        assert node["tags"] == node["domains"]

    def test_add_node_tags_supplement_domains(self, store):
        nid = store.add_node("Y", domains=["auto"], tags=["user"])
        node = store.get_node(nid)
        assert "auto" in node["domains"]
        assert "user" in node["domains"]

    def test_update_node_with_tags(self, store):
        nid = store.add_node("Z", domains=["old"])
        store.update_node(nid, tags=["new1", "new2"])
        node = store.get_node(nid)
        assert "new1" in node["domains"]
        assert "new2" in node["domains"]

    def test_row_to_dict_includes_tags(self, store):
        nid = store.add_node("T", domains=["x", "y"])
        node = store.get_node(nid)
        assert "tags" in node
        assert node["tags"] == ["x", "y"]

    def test_tag_filter_no_partial_match(self, store):
        """Tag 'ml' should not match 'html'."""
        store.add_node("A", domains=["html"])
        store.add_node("B", domains=["ml"])
        results = store.all_nodes(tags=["ml"])
        assert len(results) == 1
        assert results[0]["title"] == "B"


class TestStats:
    def test_stats(self, store):
        store.add_node("A", node_id="a")
        store.add_node("B", node_id="b")
        store.add_edge("a", "b")
        s = store.stats()
        assert s["nodes"] == 2
        assert s["edges"] >= 1
