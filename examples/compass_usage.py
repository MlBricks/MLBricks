from mlbricks import ESA, compass


def evaluate_fn(*, backend: str, c: int, precision: str):
    # Replace with your real validation + throughput benchmark.
    simulated = {
        8: (1.90, 900_000),
        16: (1.88, 1_000_000),
        32: (1.89, 1_050_000),
        64: (1.93, 1_080_000),
    }
    val_loss, tok_per_sec = simulated[c]
    return {"backend": backend, "c": c, "val_loss": val_loss, "tok_per_sec": tok_per_sec}


result = compass(evaluate_fn=evaluate_fn, c_candidates=(8, 16, 32, 64))
print(result.summary())

# Most users should simply use ESA(...), whose default is compass="auto".
manual_layer = ESA(embd=128, head=4, compass=result.recommended)
