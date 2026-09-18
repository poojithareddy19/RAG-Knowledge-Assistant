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

def test_a_track_question_routes_to_chart():
    # "float" matches DATA_WORDS, so without the domain words this came back
    # as a table of coordinates rather than a map.
    assert rule_route("show the track of float 1900083") == "chart"


def test_a_map_question_routes_to_chart():
    assert rule_route("map the floats in the Arabian Sea") == "chart"


def test_a_where_did_question_routes_to_chart():
    assert rule_route("where did float 1900083 go in 2003") == "chart"


def test_counting_profiles_is_still_a_data_question():
    assert rule_route("how many profiles are there per region") == "data"
