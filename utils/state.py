"""Identity includes full historical data and every research assumption."""

import hashlib
import json
from dataclasses import asdict

import pandas as pd

ENGINE_VERSION = "2.0.0"


def result_key(data, strategy, execution):
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(data.bars, index=True).values.tobytes()
    ).hexdigest()
    payload = {
        "engine": ENGINE_VERSION,
        "symbol": data.symbol,
        "source": data.source,
        "currency": data.currency,
        "demo": data.demo,
        "bars": digest,
        "strategy": asdict(strategy),
        "execution": asdict(execution),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
