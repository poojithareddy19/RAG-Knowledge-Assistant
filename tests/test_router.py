from src.router.classifier import ROUTES, route, rule_route


def test_the_three_routes_are_summaries_data_and_chart():
    assert ROUTES == ("summaries", "data", "chart")


def test_chart_words_win():
    assert rule_route("plot average temperature by year") == "chart"


def test_data_question_routes_to_data():
    assert (
        rule_route("what is the average surface temperature per year")
        == "data"
    )


def test_a_descriptive_question_routes_to_summaries():
    assert rule_route("tell me about float 1901393") == "summaries"
    assert rule_route("describe the Arabian Sea data") == "summaries"
    assert rule_route("what does float 1901393 measure") == "summaries"


def test_summary_words_beat_data_words():
    """"Tell me about" names a description, and "float" alone would have sent
    it to SQL, which answers with a table nobody asked for."""
    assert rule_route("tell me about the floats in the Bay of Bengal") == "summaries"


def test_asking_which_floats_carry_a_sensor_is_a_summary_question():
    """Sent to SQL, the model invented a dissolved_oxygen_qc column and the
    scope guard refused a question every BGC summary answers in one clause."""
    assert rule_route("which floats measure dissolved oxygen in the Arabian Sea") == "summaries"
    assert rule_route("floats with an oxygen sensor") == "summaries"
    # Three words of sensor name. Asked in the app, this went to SQL and
    # timed out, because the pattern allowed one word before "sensor".
    assert (
        rule_route("which floats in the Bay of Bengal carry a dissolved oxygen sensor")
        == "summaries"
    )


def test_a_chart_verb_still_beats_a_summary_phrase():
    assert rule_route("plot an overview of the temperature") == "chart"


def test_an_unplaceable_question_is_left_to_the_model():
    assert rule_route("is the water warmer near Goa") is None


def test_fallback_is_used_when_model_disabled():
    route_name, how = route(
        "is the water warmer near Goa",
        use_model=False,
    )
    assert route_name == "summaries"
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
