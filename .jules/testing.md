## 2024-08-14 - Test Configuration URL Lookup
**Learning:** Always use `unittest.mock.patch.dict` when mocking global configuration dictionaries like `config.SOURCE_URLS` rather than hardcoding the actual production URLs within the tests.
**Action:** Use `@patch.dict('module.DICT_NAME', {'test-key': 'test-value'})` to isolate tests from production configuration changes and ensure test robustness.
