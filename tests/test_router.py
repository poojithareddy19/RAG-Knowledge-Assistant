from src.router.classifier import route, rule_route


def test_chart_words_win():
    assert rule_route("plot average temperature by year") == "chart"


def test_data_question_routes_to_data():
    assert (
        rule_route("what is the average surface temperature per year")
        == "data"
    )


def test_document_words_beat_data_words():
    question = "what does the policy document say about temperature limits"
    assert rule_route(question) is None


def test_fallback_is_used_when_model_disabled():
    route_name, how = route(
        "summarise clause 4 of the handbook",
        use_model=False,
    )
    assert route_name == "documents"
    assert how == "fallback"