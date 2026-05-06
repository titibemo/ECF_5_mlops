from churnguard.train import train_model

def test_train_model_returns_dict():
    result = train_model("rf")

    assert isinstance(result, dict)
    assert "model_name" in result
    assert "metrics" in result
    assert "run_id" in result