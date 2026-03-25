def pytest_addoption(parser):
    parser.addoption(
        "--amp-test",
        action="store_true",
        default=False,
        help="Enable AMP-specific tests",
    )
