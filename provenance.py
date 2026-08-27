#!/usr/bin/env python3
"""Build a W3 PROV provenance chain for an MLflow run and export it as an image.

Given a run URL (or run id) on the MLflow server, the script:
  1. finds matching runs — same user, or runs sharing the same datasets
     (upstream artifact runs referenced by the run's parameters/tags),
  2. walks the chain upstream from the run until nothing resolves further
     (i.e. back to dataset creation),
  3. collects every parameter and every environment artifact (sha256 + size),
  4. writes the chain as W3 PROV:
       - provenance_<run_id>.prov.json   (CommonProvenanceModel JSON-LD)
       - <out-dir>/prov-<name>.png       (graph, rendered via prov.dot like the
                                          RationAI/crc_ml-provenance repo does:
                                          prov.dot.prov_to_dot(bundle).write_png)
       - <out-dir>/prov-<name>.provn     (PROV-N serialization)

Usage:
    python provenance.py RUN_URL [--out-dir DIR] [--no-env]

Default run: the test run from the task.
Self-check (offline): python provenance.py --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import mlflow
from mlflow.entities import ViewType
import prov.model
import prov.dot

DEFAULT_TRACK_URI = "https://mlflow-jiribuchta.dyn.cloud.trusted.e-infra.cz/"
DEFAULT_URL = ("https://mlflow-jiribuchta.dyn.cloud.trusted.e-infra.cz/"
              "#/experiments/1/runs/998f2e710351420db30ce182f284c321")

# artifact URIs look like mlflow-artifacts:/<exp_id>/<run_id>/artifacts/<path>
URI_RE = re.compile(r"mlflow-artifacts:/(\d+)/([0-9a-f]{32})/artifacts?/?(.*)")

PREFIX = {
    "storage": "http://localhost:8083/api/v1/documents/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "schema": "https://schema.org/",
    "cpm": "https://www.commonprovenancemodel.org/cpm-namespace-v1-0/",
    "blank": "https://openprovenance.org/blank/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "gen": "gen/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "sosa": "http://www.w3.org/ns/sosa/",
}

# namespaces registered on the prov document (same pattern as
# rationai/provenance/provenance.py::prepare_document in RationAI/crc_ml-provenance)
NAMESPACE_URIS = {
    "gen": "gen/",
    "schema": "https://schema.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "sosa": "http://www.w3.org/ns/sosa/",
}


# --- small utilities --------------------------------------------------------

def iso(ms: int | None) -> str:
    """ms epoch -> ISO-8601 UTC (PROV xsd:dateTime)."""
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:100]


def _qn(name: str) -> dict:
    """A PROV qualified name: {"type": "prov:QUALIFIED_NAME", "$": name}."""
    return {"type": "prov:QUALIFIED_NAME", "$": name}


# --- helpers for talking to MLflow from Python -----------------------------

class Probe:
    """Thin wrapper around MlflowClient with the queries this script needs."""

    def __init__(self, track_uri: str):
        mlflow.set_tracking_uri(track_uri)
        self.client = mlflow.MlflowClient()
        self._cache: dict[str, object] = {}

    # runs
    def get_run(self, run_id: str):
        if run_id not in self._cache:
            self._cache[run_id] = self.client.get_run(run_id)
        return self._cache[run_id]

    def has_run(self, run_id: str) -> bool:
        try:
            self.get_run(run_id)
            return True
        except mlflow.exceptions.MlflowException:
            return False

    def name(self, run_id: str) -> str:
        tags = self.get_run(run_id).data.tags or {}
        return tags.get("mlflow.runName", run_id[:8])

    # artifacts
    def artifacts(self, run_id: str, path: str | None = None):
        return self.client.list_artifacts(run_id, path=path)

    def walk_artifacts(self, run_id: str, path: str = ""):
        """Recursive list of (relative_path, is_dir, size)."""
        out = []
        for a in self.client.list_artifacts(run_id, path=path or None):
            p = f"{path}/{a.path}" if path else a.path
            out.append((p, a.is_dir, a.file_size))
            if a.is_dir:
                out.extend(self.walk_artifacts(run_id, p))
        return out

    def download(self, run_id: str, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            local = self.client.download_artifacts(run_id, path, d)
            return Path(local).read_bytes()

    # provenance-ish queries
    def input_refs(self, run_id: str) -> list[tuple[str, str, str]]:
        """Every upstream (exp_id, run_id, full_uri) referenced by params/tags."""
        run = self.get_run(run_id)
        found: list[tuple[str, str, str]] = []

        def scan(v):
            if isinstance(v, str):
                for m in URI_RE.finditer(v):
                    uri = (f"mlflow-artifacts:/{m.group(1)}/{m.group(2)}"
                           f"/artifacts/{m.group(3).strip('/')}")
                    found.append((m.group(1), m.group(2), uri))
            elif isinstance(v, (list, tuple)):
                for x in v:
                    scan(x)
            elif isinstance(v, dict):
                for x in v.values():
                    scan(x)

        scan(run.data.params)
        scan(run.data.tags)
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for e, r, u in found:
            if (e, r) not in seen:
                seen.add((e, r))
                out.append((e, r, u))
        return out

    def matching_runs(self, run_id: str) -> dict:
        """Runs related to run_id by creator or by shared datasets.

        Returns {'same_user': [...], 'shared_dataset': [...]} where each
        entry is (run_id, name). 'shared_dataset' = runs that consumed an
        artifact of the same upstream run this run consumed (i.e. they were
        fed by the same dataset-producing run).
        """
        run = self.get_run(run_id)
        my_upstreams = {r for _, r, _ in self.input_refs(run_id)}
        same_user: list[tuple[str, str]] = []
        shared_dataset: list[tuple[str, str]] = []
        for r in self.client.search_runs(experiment_ids=[run.info.experiment_id],
                                        run_view_type=ViewType.ALL):
            rid = r.info.run_id
            if rid == run_id:
                continue
            nm = (r.data.tags or {}).get("mlflow.runName", rid[:8])
            if r.info.user_id == run.info.user_id:
                same_user.append((rid, nm))
            if my_upstreams & {x[1] for x in self.input_refs(rid)}:
                shared_dataset.append((rid, nm))
        return {"same_user": same_user, "shared_dataset": shared_dataset}


# --- chain -------------------------------------------------------------------

def build_chain(p: Probe, root_id: str) -> dict[str, dict]:
    """Walk upstream from root_id through artifact references until every
    referenced run either exists on the server or is gone (old store).

    Returns {run_id: {'run', 'name', 'user', 'inputs': [(exp, up, uri)]}}
    in discovery order (root first).
    """
    chain: dict[str, dict] = {}
    queue = [root_id]
    while queue:
        rid = queue.pop(0)
        if rid in chain or not p.has_run(rid):
            continue
        run = p.get_run(rid)
        chain[rid] = {
            "run": run,
            "name": p.name(rid),
            "user": run.info.user_id,
            "inputs": p.input_refs(rid),
        }
        queue.extend(up for _, up, _ in chain[rid]["inputs"] if up not in chain)
    return chain


def environment_info(p: Probe, run_id: str) -> dict | None:
    """{artifact_path: {'sha256':..., 'size':...}} for the run's
    environment/ artifact dir, or None if the run has no such dir."""
    env = {pth: f for pth, is_dir, f in p.walk_artifacts(run_id, "environment")
           if not is_dir}
    if not env:
        return None
    return {pth: {"sha256": hashlib.sha256(p.download(run_id, pth)).hexdigest(),
                  "size": f}
            for pth, f in env.items()}


def collect(p: Probe, root_id: str, include_environment: bool = True):
    """(chain, envs) — everything the builders below need."""
    chain = build_chain(p, root_id)
    envs = {rid: environment_info(p, rid) for rid in chain} if include_environment else {}
    return chain, envs


# --- W3 PROV / CPM document ---------------------------------------------------

def build_provenance(p: Probe, root_id: str, include_environment: bool = True) -> dict:
    """Assemble the CPM bundle (JSON-LD dict) for the whole chain rooted at
    root_id. Full parameters and environment are embedded as prov:Annotation
    entities (via qualifiedAssociation) characterizing each activity."""
    chain, envs = collect(p, root_id, include_environment)
    entity: dict = {}
    activity: dict = {}
    agent: dict = {}
    used: dict = {}
    was_generated_by: dict = {}
    was_derived_from: dict = {}
    qualified_association: dict = {}
    was_attributed_to: dict = {}
    n = 0

    for run_id, info in chain.items():
        run = info["run"]
        info_ = run.info
        act = f"gen:run_{run_id}"
        user = info["user"]

        # agent (user) + attribution
        agent.setdefault(f"gen:user_{user}", {
            "schema:name": [user],
            "prov:type": [_qn("schema:Person")],
            "schema:affiliation": ["RationAI"],
        })
        was_attributed_to[f"_:n{n}"] = {"prov:entity": act, "prov:agent": f"gen:user_{user}"}
        n += 1

        # activity (the run itself)
        tags = run.data.tags or {}
        activity[act] = {
            "prov:type": [_qn("schema:Action")],
            "prov:startTime": [iso(info_.start_time)],
            "prov:endTime": [iso(info_.end_time)],
            "schema:name": [info["name"]],
            "schema:identifier": [run_id],
            "dct:description": [tags.get("mlflow.note.content", "")],
        }

        # all parameters -> prov:Annotation characterizing the activity
        ann_p = f"gen:annotation_params_{run_id}"
        entity[ann_p] = {
            "prov:type": [_qn("prov:Annotation")],
            "prov:annotation": [json.dumps(dict(run.data.params),
                                            ensure_ascii=False, sort_keys=True)],
        }
        qualified_association[f"_:n{n}"] = {
            "prov:activity": act,
            "prov:annotation": ann_p,
            "prov:annotatedEntity": act,
        }
        n += 1

        # environment artifacts -> second prov:Annotation
        if include_environment:
            env = envs.get(run_id)
            if env:
                ann_e = f"gen:annotation_environment_{run_id}"
                entity[ann_e] = {
                    "prov:type": [_qn("prov:Annotation")],
                    "prov:annotation": [json.dumps(env, ensure_ascii=False,
                                                    sort_keys=True)],
                }
                qualified_association[f"_:n{n}"] = {
                    "prov:activity": act,
                    "prov:annotation": ann_e,
                    "prov:annotatedEntity": act,
                }
                n += 1

        # output entity generated by this activity
        out_ent = f"gen:output_{run_id}"
        entity[out_ent] = {
            "schema:name": [info["name"]],
            "prov:type": [_qn("sosa:Sample")],
            "dct:description": [f"Output of {info['name']}"],
            "schema:url": [info_.artifact_uri],
        }
        was_generated_by[f"_:n{n}"] = {"prov:entity": out_ent, "prov:activity": act}
        n += 1

        # inputs: entity per artifact URI, used by this activity; when the
        # producing run is in the chain, link the two (wasDerivedFrom) — this
        # is what makes the chain a chain.
        for _exp, up_id, uri in info["inputs"]:
            inp = f"gen:input_{slug(uri)}"
            entity.setdefault(inp, {
                "schema:name": [uri],
                "schema:url": [uri],
                "prov:type": [_qn("sosa:Sample")],
            })
            used[f"_:n{n}"] = {"prov:activity": act, "prov:entity": inp}
            n += 1
            if up_id in chain:
                was_derived_from[f"_:n{n}"] = {
                    "prov:entity": inp,
                    "prov:derivation": f"gen:output_{up_id}",
                }
                n += 1

    # bundle metadata for the root run
    root = chain[root_id]
    meta_id = f"meta:{root_id}"
    entity[meta_id] = {
        "prov:type": [_qn("cpm:BundleMetadata")],
        "gen:run_name": [root["name"]],
        "gen:output_name": [root["name"]],
        "cpm:organization": ["RationAI"],
        "gen:input_uris": [json.dumps([u for _, _, u in root["inputs"]],
                                      ensure_ascii=False)],
    }
    main_act = f"blank:Run_{root_id[:10]}"
    activity[main_act] = {
        "prov:type": [_qn("cpm:mainActivity")],
        "cpm:referencedMetaBundleId": [_qn(meta_id)],
        "dct:hasPart": [_qn(f"gen:run_{root_id}")],
    }

    was_associated: dict = {}
    for run_id, info in chain.items():
        was_associated[f"_:n{n}"] = {
            "prov:activity": f"gen:run_{run_id}",
            "prov:agent": f"gen:user_{info['user']}",
        }
        n += 1

    return {
        "bundle": {
            f"storage:{root_id}": {
                "prefix": PREFIX,
                "entity": entity,
                "activity": activity,
                "agent": agent,
                "wasAssociatedWith": was_associated,
                "wasAttributedTo": was_attributed_to,
                "wasGeneratedBy": was_generated_by,
                "used": used,
                "wasDerivedFrom": was_derived_from,
                "qualifiedAssociation": qualified_association,
            }
        }
    }


# --- prov package document + exports (same way as RationAI/crc_ml-provenance) ---

def build_prov_document(p: Probe, root_id: str) -> prov.model.ProvDocument:
    """The chain as a prov.model.ProvDocument (one bundle), so it can be
    exported to PNG / PROV-N exactly like rationai.utils.provenance does."""
    doc = prov.model.ProvDocument()
    for prefix, uri in NAMESPACE_URIS.items():
        doc.add_namespace(prefix, uri)
    # bare attribute keys resolve against this default namespace
    doc.set_default_namespace("http://example.org/0/")
    chain = build_chain(p, root_id)
    bndl = doc.bundle(f"gen:bundle_{root_id}")

    for run_id, info in chain.items():
        run = info["run"]
        info_ = run.info
        # attributes stay small on purpose: prov.dot renders every attribute
        # (show_element_attributes=True, the repo default) into the image.
        # Full params/environment JSON lives in the CPM .prov.json instead.
        act = bndl.activity(f"gen:run_{run_id}", other_attributes={
            "schema:name": info["name"],
            "schema:identifier": run_id,
            "dct:description": (run.data.tags or {}).get("mlflow.note.content", ""),
            "prov:startTime": iso(info_.start_time),
            "prov:endTime": iso(info_.end_time),
        })
        user = info["user"]
        bndl.agent(f"gen:user_{user}", other_attributes={
            "schema:name": user,
            "schema:affiliation": "RationAI",
        })
        bndl.wasAssociatedWith(act, f"gen:user_{user}")
        bndl.wasAttributedTo(act, f"gen:user_{user}")
        out_ent = bndl.entity(f"gen:output_{run_id}", other_attributes={
            "schema:name": info["name"],
            "schema:url": info_.artifact_uri,
        })
        bndl.wasGeneratedBy(out_ent, act)
        for _exp, up_id, uri in info["inputs"]:
            inp = bndl.entity(f"gen:input_{slug(uri)}", other_attributes={
                "schema:name": uri,
                "schema:url": uri,
            })
            bndl.used(act, inp)
            if up_id in chain:
                bndl.wasDerivedFrom(inp, f"gen:output_{up_id}")
    return doc


def export_provenance_png(bundle, name: str, out_dir: str | Path = "output") -> Path:
    """Create <out-dir>/prov-<name>.png from a bundle — the same two-liner
    as rationai.utils.provenance.export_to_image in RationAI/crc_ml-provenance:
    dot = prov.dot.prov_to_dot(bundle); dot.write_png(f"prov-{name}.png")
    """
    dot = prov.dot.prov_to_dot(bundle)
    path = Path(out_dir) / f"prov-{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    dot.write_png(str(path))
    return path


def export_provenance_provn(doc: prov.model.ProvDocument, name: str,
                            out_dir: str | Path = "output") -> Path:
    """Save the bundle as PROV-N — same as rationai.utils.provenance.export_to_provn."""
    path = Path(out_dir) / f"prov-{name}.provn"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        doc.serialize(f, format="provn")
    return path


def parse_run_id(url_or_id: str) -> str:
    m = re.search(r"[0-9a-f]{32}", url_or_id)
    if not m:
        raise ValueError(f"no 32-char hex run id in: {url_or_id!r}")
    return m.group(0)


# --- CLI ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W3 PROV chain + graph for an MLflow run")
    ap.add_argument("url", nargs="?", default=DEFAULT_URL, help="run URL or run id")
    ap.add_argument("--out-dir", default="output",
                    help="folder for the .png/.provn exports (default: output)")
    ap.add_argument("--json-out", help="CPM JSON path (default: provenance_<run_id>.prov.json)")
    ap.add_argument("--track-uri", default=DEFAULT_TRACK_URI)
    ap.add_argument("--no-env", action="store_true",
                    help="skip downloading environment artifacts")
    ap.add_argument("--self-check", action="store_true",
                    help="run the offline self-check and exit")
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()

    run_id = parse_run_id(args.url)
    p = Probe(args.track_uri)
    if not p.has_run(run_id):
        sys.exit(f"run {run_id} not found on {args.track_uri}")

    cpm = build_provenance(p, run_id, include_environment=not args.no_env)
    json_out = Path(args.json_out) if args.json_out else Path(f"provenance_{run_id}.prov.json")
    json_out.write_text(json.dumps(cpm, indent=2, ensure_ascii=False) + "\n")

    # graph exports, same way as RationAI/crc_ml-provenance
    name = slug(p.name(run_id))
    doc = build_prov_document(p, run_id)
    bundle = next(iter(doc.bundles))
    png = export_provenance_png(bundle, name, args.out_dir)
    provn = export_provenance_provn(doc, name, args.out_dir)

    # human-readable summary
    matches = p.matching_runs(run_id)
    run = p.get_run(run_id)
    print(f"Run:          {p.name(run_id)}  ({run_id})")
    print(f"User:         {run.info.user_id}")
    print(f"Started:      {iso(run.info.start_time)}")
    print(f"Matching:     {len(matches['same_user'])} runs by same user, "
          f"{len(matches['shared_dataset'])} sharing a dataset")
    chain = build_chain(p, run_id)
    print(f"Chain:        {len(chain)} runs, dataset creation -> this run")
    for rid, info in chain.items():
        upstreams = [up for _, up, _ in info["inputs"] if up != rid]
        mark = "  [root]" if not any(p.has_run(u) for u in upstreams) else ""
        print(f"  - {info['name']}  ({rid}) user={info['user']}{mark}")
    print(f"Provenance:   {json_out.resolve()}")
    print(f"Graph:        {png.resolve()}")
    print(f"PROV-N:       {provn.resolve()}")
    return 0


# ponytail: self-check is offline by design (no server, no graphviz); live
# behaviour verified by running the default URL.
def self_check() -> int:
    assert parse_run_id(DEFAULT_URL) == "998f2e710351420db30ce182f284c321"
    assert parse_run_id("998f2e710351420db30ce182f284c321") == "998f2e710351420db30ce182f284c321"
    m = URI_RE.search("x mlflow-artifacts:/111/e8175ecf823d403ca5b629a9bb3cf874/artifacts/train.csv y")
    assert m.group(1) == "111" and m.group(2) == "e8175ecf823d403ca5b629a9bb3cf874"
    m = URI_RE.search("mlflow-artifacts:/1/" + "ab" * 16 + "/artifacts/embeddings/train")
    assert m.group(2) == "ab" * 16 and m.group(3).strip() == "embeddings/train"
    assert URI_RE.search("no artifact here") is None
    assert slug("🏋️‍♂️ Training - high - vgg16") == "training_high_vgg16"
    assert iso(None) == "" and iso(0) == "1970-01-01T00:00:00+00:00"

    # a stub Probe must produce a CPM bundle AND a prov document with every
    # expected section / record type
    rid_a, rid_b = "a" * 32, "b" * 32

    class FakeRun:
        class info:
            run_id, user_id, experiment_id = rid_a, "u1", "1"
            start_time, end_time = 1700000000000, 1700000060000
            artifact_uri = f"mlflow-artifacts:/1/{rid_a}/artifacts"

        class data:
            params = {"lr": "0.001"}
            tags = {"mlflow.runName": "Root Train"}
            metrics = {}

    stub = SimpleNamespace(
        get_run=lambda rid: FakeRun(),
        has_run=lambda rid: rid in (rid_a, rid_b),
        name=lambda rid: "Root Train" if rid == rid_a else "Upstream Dataset",
        input_refs=lambda rid: (
            [("1", rid_b, f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv")]
            if rid == rid_a else []
        ),
        walk_artifacts=lambda rid, path=None: [],
        download=lambda rid, path: b"",
    )

    # CPM dict
    cpm = build_provenance(stub, rid_a)
    b = cpm["bundle"][f"storage:{rid_a}"]
    for key in ("prefix", "entity", "activity", "agent", "wasAssociatedWith",
                "wasAttributedTo", "wasGeneratedBy", "used", "wasDerivedFrom",
                "qualifiedAssociation"):
        assert key in b and b[key], f"missing/empty section {key}"
    assert f"gen:run_{rid_a}" in b["activity"] and f"gen:run_{rid_b}" in b["activity"]
    assert any("gen:input_" in k for k in b["entity"])
    assert any("gen:output_" in k for k in b["entity"])
    assert any("gen:annotation_params_" in k for k in b["entity"])
    assert any(k == f"gen:user_u1" for k in b["agent"])
    assert any(d.get("prov:derivation") == f"gen:output_{rid_b}"
               for d in b["wasDerivedFrom"].values())

    # prov document (same object model RationAI/crc_ml-provenance exports)
    doc = build_prov_document(stub, rid_a)
    assert len(doc.bundles) == 1
    bundle = next(iter(doc.bundles))
    records = bundle.get_records()
    types = [type(r).__name__ for r in records]
    # prov record classes: ProvGeneration=wasGeneratedBy, ProvUsage=used,
    # ProvDerivation=wasDerivedFrom, ProvAssociation=wasAssociatedWith,
    # ProvAttribution=wasAttributedTo
    for expected in ("ProvGeneration", "ProvUsage", "ProvDerivation",
                     "ProvAssociation", "ProvAttribution"):
        assert expected in types, f"missing record {expected}"
    ents = [r for r in bundle.get_records() if isinstance(r, prov.model.ProvElement)]
    ids = {str(e.identifier): e for e in ents}  # QualifiedName -> "gen:xxx"
    assert f"gen:run_{rid_a}" in ids and f"gen:run_{rid_b}" in ids
    inp = f"gen:input_" + slug(f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv")
    assert inp in ids and any(
        v.endswith("dataset.csv")
        for v in ids[inp].get_attribute("schema:name") or [])
    print("self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
