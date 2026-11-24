from synthetic_traffic_validation.verify_evidence import verify


def test_archived_tables_match_the_supplied_pdf():
    evidence = verify()
    assert evidence["policy_records"] == 2728
    assert evidence["optimizer_attempts"] == 998
    assert evidence["pdf_page_8_rows"] == 6
