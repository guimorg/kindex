"""Tests for Kindex MCP server tool functions."""

import json
import os

import pytest

mcp = pytest.importorskip("mcp", reason="mcp not installed")


@pytest.fixture
def mcp_store(tmp_path):
    """Set up a Store + Config for MCP tool testing."""
    from kindex.config import load_config
    from kindex.store import Store

    d = str(tmp_path)
    cfg = load_config()
    cfg.data_dir = d
    store = Store(cfg)

    # Add test data
    id1 = store.add_node(title="Stigmergy", content="Coordination through environmental traces",
                         node_type="concept", domains=["biology", "ai"],
                         prov_activity="test")
    id2 = store.add_node(title="Python", content="Expert-level programming skill",
                         node_type="skill", domains=["engineering"],
                         prov_activity="test")
    store.add_node(title="Never break the API contract",
                   content="All public endpoints must maintain backward compatibility",
                   node_type="constraint", prov_activity="test",
                   extra={"trigger": "pre-deploy", "action": "block"})
    store.add_edge(id1, id2, edge_type="relates_to", weight=0.6)

    yield store, cfg
    store.close()


@pytest.fixture
def patch_store(mcp_store, monkeypatch):
    """Patch the MCP server's _get_store to use test store."""
    store, cfg = mcp_store
    import kindex.mcp_server as mcp_mod
    monkeypatch.setattr(mcp_mod, "_store", store)
    monkeypatch.setattr(mcp_mod, "_config", cfg)
    return store, cfg


class TestMCPSearch:
    def test_search_finds_results(self, patch_store):
        from kindex.mcp_server import search
        result = search("stigmergy")
        assert "Stigmergy" in result
        assert "stigmergy" in result.lower()

    def test_search_no_results(self, patch_store):
        from kindex.mcp_server import search
        result = search("zzzznonexistent")
        assert "No results" in result or "0 results" in result or "Found" in result


class TestMCPAdd:
    def test_add_creates_node(self, patch_store):
        from kindex.mcp_server import add
        result = add("Graph theory is about vertices and edges", node_type="concept")
        assert "Created node" in result

    def test_add_with_type(self, patch_store):
        from kindex.mcp_server import add
        result = add("Should we use Redis?", node_type="question")
        assert "Created node" in result
        assert "question" in result


class TestMCPContext:
    def test_context_with_topic(self, patch_store):
        from kindex.mcp_server import context
        result = context(topic="stigmergy", level="abridged")
        assert "Kindex" in result or "stigmergy" in result.lower()

    def test_context_empty_topic(self, patch_store):
        from kindex.mcp_server import context
        result = context()
        # Should return something (recent nodes fallback)
        assert isinstance(result, str)


class TestMCPShow:
    def test_show_by_title(self, patch_store):
        from kindex.mcp_server import show
        result = show("Stigmergy")
        assert "Stigmergy" in result
        assert "concept" in result

    def test_show_not_found(self, patch_store):
        from kindex.mcp_server import show
        result = show("nonexistent-node-id")
        assert "not found" in result.lower()


class TestMCPLink:
    def test_link_nodes(self, patch_store):
        from kindex.mcp_server import add, link
        add("Machine Learning fundamentals", node_type="concept")
        result = link("Python", "Machine Learning fundamentals",
                       relationship="implements", weight=0.8)
        assert "Linked" in result

    def test_link_not_found(self, patch_store):
        from kindex.mcp_server import link
        result = link("nonexistent", "also-nonexistent")
        assert "not found" in result.lower()


class TestMCPListNodes:
    def test_list_all(self, patch_store):
        from kindex.mcp_server import list_nodes
        result = list_nodes()
        assert "node(s)" in result
        assert "Stigmergy" in result

    def test_list_by_type(self, patch_store):
        from kindex.mcp_server import list_nodes
        result = list_nodes(node_type="skill")
        assert "Python" in result

    def test_list_empty(self, patch_store):
        from kindex.mcp_server import list_nodes
        result = list_nodes(node_type="nonexistent-type")
        assert "No nodes" in result

    def test_list_uses_canonical_store_page(self, patch_store, monkeypatch):
        store, _ = patch_store
        calls = []

        def page_nodes(**kwargs):
            calls.append(kwargs)
            return [], 1

        monkeypatch.setattr(store, "page_nodes", page_nodes)
        monkeypatch.setattr(
            store,
            "count_nodes",
            lambda **kwargs: pytest.fail("MCP must not count outside Store.page_nodes"),
        )
        monkeypatch.setattr(
            store,
            "all_nodes",
            lambda **kwargs: pytest.fail("MCP must not list outside Store.page_nodes"),
        )

        from kindex.mcp_server import list_nodes

        result = list_nodes(node_type="concept", limit=1, offset=2)

        assert "0 node(s) returned of 1 matching" in result
        assert calls == [
            {
                "node_type": "concept",
                "status": None,
                "audience": None,
                "tags": None,
                "since": None,
                "until": None,
                "limit": 1,
                "offset": 2,
                "order": "weight",
            }
        ]

    def test_list_temporal_page_includes_total_next_offset_and_time(self, patch_store):
        store, _ = patch_store
        store.add_node("Day one", node_id="day-one", prov_when="2030-01-01T12:00:00Z")
        store.add_node("Day two", node_id="day-two", prov_when="2030-01-02T12:00:00Z")

        from kindex.mcp_server import list_nodes

        result = list_nodes(
            since="2030-01-01",
            until="2030-01-03",
            limit=1,
            offset=0,
            order="event_time_asc",
        )

        assert "1 node(s) returned of 2 matching" in result
        assert "Next offset: 1" in result
        assert "Day one" in result
        assert "when=2030-01-01T12:00:00.000Z" in result

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"since": "not-a-date"}, "since must be an ISO"),
            ({"until": "not-a-date"}, "until must be an ISO"),
            (
                {"since": "2030-01-03", "until": "2030-01-02"},
                "since must be earlier than or equal to until",
            ),
            ({"offset": -1}, "offset must be non-negative"),
            ({"limit": 0}, "limit must be positive"),
            ({"order": "newest"}, "order must be"),
        ],
    )
    def test_list_rejects_invalid_temporal_parameters(self, patch_store, kwargs, message):
        from kindex.mcp_server import list_nodes

        assert message in list_nodes(**kwargs)

    def test_list_allows_offset_beyond_total(self, patch_store):
        from kindex.mcp_server import list_nodes

        result = list_nodes(offset=10_000)

        assert "0 node(s) returned" in result
        assert "offset=10000" in result
        assert "Next offset" not in result


class TestMCPStatus:
    def test_status_returns_stats(self, patch_store):
        from kindex.mcp_server import status
        result = status()
        assert "Nodes:" in result
        assert "Edges:" in result


class TestMCPAsk:
    def test_ask_procedural(self, patch_store):
        from kindex.mcp_server import ask
        result = ask("How do I use stigmergy?")
        assert "procedural" in result.lower()

    def test_ask_factual(self, patch_store):
        from kindex.mcp_server import ask
        result = ask("What is Python?")
        assert "factual" in result.lower()

    def test_ask_decision(self, patch_store):
        from kindex.mcp_server import ask
        result = ask("Should I use stigmergy vs direct communication?")
        assert "decision" in result.lower()


class TestMCPSuggest:
    def test_suggest_empty(self, patch_store):
        from kindex.mcp_server import suggest
        result = suggest()
        assert "No pending" in result or "suggestion" in result.lower()


class TestMCPLearn:
    def test_learn_extracts(self, patch_store):
        from kindex.mcp_server import learn
        result = learn("We decided to use Redis for caching because of its speed. "
                       "This connects to our Distributed Systems architecture.")
        assert "Extracted" in result


class TestMCPGraphStats:
    def test_graph_stats(self, patch_store):
        from kindex.mcp_server import graph_stats
        result = graph_stats()
        assert "Nodes:" in result
        assert "Density:" in result


class TestMCPChangelog:
    def test_changelog(self, patch_store):
        from kindex.mcp_server import changelog
        result = changelog(days=30)
        assert isinstance(result, str)


class TestMCPResources:
    def test_resource_status(self, patch_store):
        from kindex.mcp_server import resource_status
        result = resource_status()
        data = json.loads(result)
        assert "nodes" in data

    def test_resource_node(self, patch_store):
        from kindex.mcp_server import resource_node
        result = resource_node("Stigmergy")
        assert "Stigmergy" in result

    def test_resource_recent(self, patch_store):
        from kindex.mcp_server import resource_recent
        result = resource_recent()
        assert isinstance(result, str)

    def test_resource_orphans(self, patch_store):
        from kindex.mcp_server import resource_orphans
        result = resource_orphans()
        assert isinstance(result, str)


class TestMCPTags:
    def test_add_with_tags(self, patch_store):
        from kindex.mcp_server import add
        result = add("Tagged concept for testing", tags="python,ml")
        assert "Created node" in result
        store, _ = patch_store
        node = store.get_node_by_title("Tagged concept for testing")
        assert "python" in node["domains"]
        assert "ml" in node["domains"]

    def test_add_tags_supplement_domains(self, patch_store):
        from kindex.mcp_server import add
        result = add("Dual tagged item", tags="user-tag", domains="auto-tag")
        assert "Created node" in result
        store, _ = patch_store
        node = store.get_node_by_title("Dual tagged item")
        assert "user-tag" in node["domains"]
        assert "auto-tag" in node["domains"]

    def test_search_with_tags_filter(self, patch_store):
        from kindex.mcp_server import search
        result = search("coordination", tags="biology")
        assert "Stigmergy" in result

    def test_search_with_tags_excludes_non_matching(self, patch_store):
        from kindex.mcp_server import search
        result = search("coordination", tags="nonexistent")
        assert "No results" in result

    def test_list_nodes_with_tags(self, patch_store):
        from kindex.mcp_server import list_nodes
        result = list_nodes(tags="biology")
        assert "Stigmergy" in result

    def test_list_nodes_default_limit(self, patch_store):
        import inspect
        from kindex.mcp_server import list_nodes
        sig = inspect.signature(list_nodes)
        assert sig.parameters["limit"].default == 100
