import json
import os
import pytest

from suggest import identify_worst_pages, build_judge_prompt

def test_identify_worst_pages():
    run_data = {
        "config": {"model": "test"},
        "pages": [
            {"page_id": "p1", "metrics": {"clustering_f1": 0.8}},
            {"page_id": "p2", "metrics": {"clustering_f1": 0.5}},
            {"page_id": "p3", "metrics": {"clustering_f1": 0.9}},
            {"page_id": "p4", "metrics": {"clustering_f1": 0.6}},
        ]
    }
    worst = identify_worst_pages(run_data, top_k=2)
    assert len(worst) == 2
    assert worst[0]["page_id"] == "p2"
    assert worst[1]["page_id"] == "p4"


def test_build_judge_prompt(tmp_path, monkeypatch):
    # Mock the interim directory to provide fragment text
    interim_dir = tmp_path / "data" / "1_interim" / "fragments"
    interim_dir.mkdir(parents=True)
    
    frag_data = [{"id": "r_1", "text": "Hello"}, {"id": "r_2", "text": "World"}]
    with open(interim_dir / "p2.json", "w") as f:
        json.dump(frag_data, f)
        
    # Override os.path.join inside suggest.py specifically for the fragment cache path
    original_join = os.path.join
    def mock_join(*args):
        if args[:3] == ("data", "1_interim", "fragments"):
            return str(interim_dir / args[3])
        return original_join(*args)
        
    monkeypatch.setattr(os.path, "join", mock_join)
    
    run_data = {
        "config": {
            "system_prompt": "Test System",
            "user_prompt_template": "Test User"
        }
    }
    worst_pages = [
        {
            "page_id": "p2",
            "metrics": {"clustering_f1": 0.5},
            "ground_truth_items": [{"class": "article", "fragment_ids": ["r_1", "r_2"]}],
            "predicted_items": [{"class": "article", "fragment_ids": ["r_1"]}, {"class": "article", "fragment_ids": ["r_2"]}],
        }
    ]
    
    prompt = build_judge_prompt(run_data, worst_pages)
    
    assert "Test System" in prompt
    assert "Test User" in prompt
    assert "Page: p2" in prompt
    assert "r_1: 'Hello" in prompt
    assert "r_2: 'World" in prompt
