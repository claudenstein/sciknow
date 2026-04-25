"""sciknow v2 — llama-server inference substrate.

Subprocess lifecycle (server.py) + HTTP clients (client.py) for the
three logical roles: writer / embedder / reranker. Replaces the v1
in-process model loads (FlagEmbedding, sentence-transformers) and the
ollama-python client.

Public surfaces:
    sciknow.infer.client.chat_stream(...)        → iterator[str]
    sciknow.infer.client.chat_complete(...)      → str
    sciknow.infer.client.embed(texts)            → list[list[float]]
    sciknow.infer.client.rerank(query, docs)     → list[float]
    sciknow.infer.server.up(role)                → start managed process
    sciknow.infer.server.down(role)              → stop
    sciknow.infer.server.status()                → list[ProcInfo]
"""

from sciknow.infer import client, server  # noqa: F401
