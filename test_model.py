import torch

from src.model import LCNN, MFM


def test_mfm():
    mfm = MFM()

    input_tensor = torch.tensor(
        [
            [1.0, 3.0, 2.0, 0.0],
        ]
    )

    output = mfm(input_tensor)

    expected = torch.tensor(
        [
            [2.0, 3.0],
        ]
    )

    print("MFM input:", input_tensor)
    print("MFM output:", output)

    assert output.shape == (1, 2)
    assert torch.equal(output, expected)


def test_lcnn():
    torch.manual_seed(42)

    model = LCNN(
        n_class=2,
        input_height=257,
        input_width=750,
        dropout=0.75,
    )

    # BatchNorm with batch_size=1 requires eval mode.
    model.eval()

    input_tensor = torch.randn(
        1,
        1,
        257,
        750,
    )

    with torch.no_grad():
        output = model(data_object=input_tensor)

    logits = output["logits"]

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nLCNN input shape:", input_tensor.shape)
    print("LCNN output keys:", output.keys())
    print("Logits shape:", logits.shape)
    print("Number of parameters:", parameter_count)

    assert "logits" in output
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def main():
    test_mfm()
    test_lcnn()

    print("\nModel smoke test passed.")


if __name__ == "__main__":
    main()