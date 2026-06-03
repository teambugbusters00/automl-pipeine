from autohf.agents.task_agent import detect_task, TaskAgent


def test_keyword_detection():
    # Exact keyword mapping
    res = detect_task("sentiment analysis of reviews", router="keyword")
    assert res.task_type == "text-classification"
    assert res.confidence == 1.0

    # Another one
    res = detect_task("named entity recognition for medical text", router="keyword")
    assert res.task_type == "token-classification"
    assert res.confidence == 1.0


def test_fuzzy_fallback():
    # Fuzzy match
    res = detect_task("extract entities", router="keyword")
    assert res.task_type == "token-classification"
    assert res.confidence >= 0.65


def test_task_agent_history():
    agent = TaskAgent(router="keyword")
    assert len(agent.history) == 0
    
    agent("sentiment of movie reviews")
    assert len(agent.history) == 1
    assert agent.history[0].task_type == "text-classification"
    
    agent.reset()
    assert len(agent.history) == 0


def test_gemma_router(monkeypatch):
    import json
    import sys
    from unittest.mock import MagicMock
    
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    mock_processor.from_pretrained.return_value = mock_processor
    mock_model.from_pretrained.return_value = mock_model
    
    # Mock processor call output
    mock_inputs = MagicMock()
    mock_inputs.get.return_value = mock_inputs
    mock_inputs.shape = (1, 10)
    mock_processor.return_value = mock_inputs
    
    # Mock model generate return
    mock_model.generate.return_value = [[1, 2, 3]]
    
    # Mock decode output JSON
    mock_output_json = json.dumps({
        "task_type": "text-classification",
        "task_label": "Text Classification",
        "keywords": ["sentiment", "classification"],
        "confidence": 0.95,
        "problem_type": "auto"
    })
    mock_processor.batch_decode.return_value = [mock_output_json]
    mock_processor.apply_chat_template.return_value = "formatted_prompt"
    
    # Mock transformers
    transformers_mock = MagicMock()
    transformers_mock.AutoProcessor = mock_processor
    transformers_mock.AutoModelForImageTextToText = mock_model
    monkeypatch.setitem(sys.modules, "transformers", transformers_mock)
    
    # Mock torch
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    
    # Run gemma router
    res = detect_task("Build an AI that detects fake reviews", router="gemma")
    
    assert res.task_type == "text-classification"
    assert res.confidence == 0.95
    assert "sentiment" in res.keywords


if __name__ == "__main__":
    test_keyword_detection()
    test_fuzzy_fallback()
    test_task_agent_history()
    # We run the mocked gemma test using a simple monkeypatch context
    class MonkeyPatch:
        def __init__(self):
            self.originals = {}
        def setitem(self, d, k, v):
            self.originals[(id(d), k)] = (d, d.get(k, None) if hasattr(d, "get") else None)
            d[k] = v
        def undo(self):
            for (d_id, k), (d, val) in self.originals.items():
                if val is None:
                    d.pop(k, None)
                else:
                    d[k] = val
                    
    mp = MonkeyPatch()
    try:
        test_gemma_router(mp)
        print("Gemma router mock test passed!")
    finally:
        mp.undo()
        
    print("All task agent tests passed!")

