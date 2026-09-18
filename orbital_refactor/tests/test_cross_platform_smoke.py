from examples.run_cross_platform_smoke import run_smoke


def test_cross_platform_smoke_uses_portable_public_contract():
    payload = run_smoke()
    assert payload["ok"] is True
    assert payload["api"]["public_api_version"] == "v1.0"
    assert payload["result_fingerprint"]["status"] == "OK"
    assert len(payload["result_fingerprint"]["position_m"]) == 3
