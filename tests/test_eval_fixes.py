"""Checks the evaluation fixes, with no GPU or model weights. Run from the repository root: `pytest tests/`.

The real eval loop (vizwiz_vqa_all_fixed_object.main) runs on three tiny synthetic images. The model call is a stub
that records the prompt and returns the pixel color at (7, 7), so the output shows *which image* each condition
was scored on. Needs: numpy, pillow, matplotlib, tqdm, pytest.
"""
import ast
import importlib.util
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

UPSTREAM = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")

COLORS = {"a": (200, 30, 30), "b": (30, 200, 30), "c": (30, 30, 200)}
CORNER = (7, 7)


# ---------- helpers ----------

def _load_defs(path, names, class_name=None, extra_globals=None):
    """Exec only the named top-level (or class-level) definitions from a file, skipping its heavy imports."""
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if class_name:
        cls = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
        body = cls.body
    keep = [n for n in body if getattr(n, "name", None) in names
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    if class_name:
        keep = [ast.ClassDef(name=class_name, bases=[], keywords=[], body=keep, decorator_list=[])]
    mod = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))
    g = dict(extra_globals or {})
    exec(compile(mod, str(path), "exec"), g)
    return g


def _closest(rgb):
    """JPEG shifts solid colors slightly; map a pixel back to a color name (or 'black')."""
    named = dict(COLORS, black=(0, 0, 0))
    return min(named, key=lambda k: sum((x - y) ** 2 for x, y in zip(named[k], rgb)))


# ---------- prompt construction (utils.py) ----------

def test_yes_no_prompt_single_question_mark():
    g = _load_defs(UPSTREAM / "utils.py", {"yes_no_prompt"})
    f = g["yes_no_prompt"]
    assert f("Is there a mouse in this image?") == "Is there a mouse in this image? Answer yes or no."
    assert f("Is this aligned") == "Is this aligned? Answer yes or no."


# ---------- category list (agents/qwen.py) ----------

class _FakeProcessor:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.prompts.append(messages[-1]["content"][-1]["text"])
        return ""


class _FakeLLM:
    def generate(self, *a, **k):
        out = types.SimpleNamespace(text="other")
        return [types.SimpleNamespace(outputs=[out])]


@pytest.fixture
def labeler():
    g = _load_defs(UPSTREAM / "agents" / "qwen.py",
                   {"METACATEGORY_ALIASES", "canonical_metacategory", "guided_labeling_per_match_per_metacategory"},
                   class_name="QWEN_VLM", extra_globals={"process_vision_info": lambda m: (None, None)})
    q = g["QWEN_VLM"].__new__(g["QWEN_VLM"])
    q.unique_categories_per_metacategory = json.loads((UPSTREAM / "categories" / "label2item_list.json").read_text())
    q.processor, q.llm, q.sampling_params = _FakeProcessor(), _FakeLLM(), None
    return q


def test_category_prompt_does_not_grow(labeler):
    before = list(labeler.unique_categories_per_metacategory["credit or debit card"]["contained_info"])
    for _ in range(3):
        labeler.guided_labeling_per_match_per_metacategory(None, "4111", "credit or debit card")
    assert len(set(labeler.processor.prompts)) == 1, "prompt changed between calls"
    assert labeler.unique_categories_per_metacategory["credit or debit card"]["contained_info"] == before


def test_metacategory_canonicalization(labeler):
    assert labeler.canonical_metacategory("Preganancy test.") == "pregnancy test"
    assert labeler.canonical_metacategory(" Credit or debit card ") == "credit or debit card"
    with pytest.raises(KeyError):
        labeler.canonical_metacategory("email")  # in the detector's list, but has no PII list


# ---------- eval loop (vizwiz_vqa_all_fixed_object.py) ----------

def _make_fixture_dir(root):
    """Three images: a and c have labeled text spans; b has none (as when labeling failed for it)."""
    (root / "imgs").mkdir()
    for d in ["results_qwen_72B_img_categories", "results_qwen_72B_augmented", "results_qwen_72B_labels",
              "hot3d_dataset/object_models"]:
        (root / d).mkdir(parents=True)
    locs, doc_labels = {}, {}
    # Span "VISA" covers the corner pixel and is labeled "Other." -- masked only under exact string matching.
    corner_poly = [[6, 6], [8, 6], [8, 8], [6, 8]]
    number_poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
    for name, rgb in COLORS.items():
        p = f"imgs/{name}.jpeg"
        Image.new("RGB", (8, 8), rgb).save(root / p, quality=100)
        locs[p] = {"full_image_mask": "8x8", "label": "credit or debit card", "object_position": [0, 0, 0]}
        (root / f"results_qwen_72B_augmented/{name}.json").write_text(json.dumps(
            {"original_detailed_data": [{"data_vlm": [[number_poly, corner_poly], ["4111 1111", "VISA"]]}]}))
        (root / f"results_qwen_72B_labels/{name}.json").write_text("{}")
        if name != "b":
            doc_labels[p] = {"finegrained_labels": {"4111 1111": "credit card number", "VISA": "Other."}}
    cat = root / "results_qwen_72B_img_categories"
    (cat / "all_meta_categories_locations_v4.json").write_text(json.dumps(locs))
    (cat / "documents_fine_grained_labels_per_metacategory.json").write_text(json.dumps(doc_labels))
    (cat / "creditcards_fine_grained_labels_per_metacategory.json").write_text("{}")
    with open(cat / "3d_objects.pkl", "wb") as f:
        pickle.dump({"hot3d_dataset/object_models/obj_000031.glb":
                     {"rendered_image": Image.new("RGBA", (2, 2), (0, 0, 0, 0))}}, f)
    (root / "hot3d_dataset/object_models/models_info.json").write_text(json.dumps({"31": {"name": "mouse"}}))


def _run_eval(tmp_path, monkeypatch, legacy_args):
    _make_fixture_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []

    def vqa_yes_prob(model_7b, processor_7b, image, question_prompt, model="qwen25", device="cuda",
                     legacy_qmark=False):
        calls.append({"prompt": question_prompt, "legacy_qmark": legacy_qmark})
        return [image.convert("RGB").getpixel(CORNER), "stub"]

    fake_utils = types.ModuleType("utils")
    fake_utils.vqa_yes_prob = vqa_yes_prob
    fake_utils.qwen25 = fake_utils.llava16 = fake_utils.gemma3 = lambda: (None, None)
    fake_utils.base64_str_to_image2 = lambda s: np.zeros(tuple(int(x) for x in s.split("x")), np.uint8)
    fake_utils.full_image_zero_outside_mask = lambda path, mask: Image.open(path).copy()
    fake_utils.NumpyEncoder = json.JSONEncoder
    for name, mod in {"utils": fake_utils, "torch": types.ModuleType("torch"), "cv2": types.ModuleType("cv2"),
                      "qwen_vl_utils": types.SimpleNamespace(process_vision_info=None)}.items():
        monkeypatch.setitem(sys.modules, name, mod)

    spec = importlib.util.spec_from_file_location("eval_under_test", UPSTREAM / "vizwiz_vqa_all_fixed_object.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["eval"] + legacy_args)
    mod.main()

    out = next(tmp_path.glob("results_qwen25_7B_img_categories_v2*/*.json"))
    return out.parent.name, json.loads(out.read_text()), calls


def test_eval_paper_behavior_reproduced(tmp_path, monkeypatch):
    out_dir, res, calls = _run_eval(tmp_path, monkeypatch, ["--legacy", "all"])
    assert out_dir == "results_qwen25_7B_img_categories_v2"
    a, b = res["imgs/a.jpeg"], res["imgs/b.jpeg"]
    # Exact matching treats "Other." as high-risk, so the corner span gets masked.
    assert _closest(a["high_risk_fine_grained_masked_private_obj"][0]) == "black"
    # Image b has no labels, yet gets a high-risk score -- computed on image a's masked image.
    assert b.get("high_risk_mask_from_previous_image") is True
    assert b["high_risk_fine_grained_masked_private_obj"][0] == a["high_risk_fine_grained_masked_private_obj"][0]
    answerability = [c["prompt"] for c in calls if "Given this question" in c["prompt"]]
    assert answerability and all(p.startswith("Is there a Given this question") for p in answerability)
    assert all(c["legacy_qmark"] for c in calls)


def test_eval_fixed_behavior(tmp_path, monkeypatch):
    out_dir, res, calls = _run_eval(tmp_path, monkeypatch, [])
    assert out_dir == "results_qwen25_7B_img_categories_v2_fixed"
    a, b, c = res["imgs/a.jpeg"], res["imgs/b.jpeg"], res["imgs/c.jpeg"]
    assert _closest(a["high_risk_fine_grained_masked_private_obj"][0]) == "a"   # "Other." span left visible
    assert b["high_risk_fine_grained_masked_private_obj"] == "None"             # excluded, not borrowed
    assert "high_risk_mask_from_previous_image" not in b
    assert _closest(c["high_risk_fine_grained_masked_private_obj"][0]) == "c"
    answerability = [c["prompt"] for c in calls if "Given this question" in c["prompt"]]
    assert answerability and all(p.startswith("Given this question") for p in answerability)
    assert not any(c["legacy_qmark"] for c in calls)


def test_eval_single_fix_toggle(tmp_path, monkeypatch):
    out_dir, res, _ = _run_eval(tmp_path, monkeypatch, ["--legacy", "stale_high_risk"])
    assert out_dir == "results_qwen25_7B_img_categories_v2_legacy-stale_high_risk"
    a, b = res["imgs/a.jpeg"], res["imgs/b.jpeg"]
    assert _closest(b["high_risk_fine_grained_masked_private_obj"][0]) == "a"
