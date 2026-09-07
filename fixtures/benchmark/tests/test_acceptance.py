from benchmark_app.export import export_json
from benchmark_app.search import search_notes
from benchmark_app.slug import slugify
from benchmark_app.store import NoteStore


def test_complete_feature_chain() -> None:
    assert slugify("  Local Agent Tea!  ") == "local-agent-tea"
    store = NoteStore()
    first = store.add("Local Agent Tea", "adaptive admission", tags=["local", "agents"])
    store.add("Other", "fixed scheduling", tags=["baseline"])
    assert search_notes(store.all(), "ADAPTIVE") == [first]
    assert search_notes(store.all(), "local") == [first]
    assert export_json(store.all()) == (
        '[{"body":"adaptive admission","slug":"local-agent-tea",'
        '"tags":["agents","local"],"title":"Local Agent Tea"},'
        '{"body":"fixed scheduling","slug":"other","tags":["baseline"],"title":"Other"}]'
    )
