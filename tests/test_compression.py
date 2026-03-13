from cuba_search.compression import split_sentences

def test_split_sentences_basic():
    text = "This is a sentence. This is another one."
    expected = ["This is a sentence.", "This is another one."]
    assert split_sentences(text) == expected

def test_split_sentences_different_punctuation():
    text = "Is this a question? Yes, it is! Wow, look at that."
    expected = ["Is this a question?", "Yes, it is!", "Wow, look at that."]
    assert split_sentences(text) == expected

def test_split_sentences_short_filtering():
    text = "Short. This is long enough. Too short."
    # "Short." is 6 chars. "This is long enough." is 20 chars. "Too short." is 10 chars.
    # Current code: if len(s.strip()) > 10
    expected = ["This is long enough."]
    assert split_sentences(text) == expected

def test_split_sentences_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("\n\t\n") == []

def test_split_sentences_non_ascii():
    text = "What about non-ASCII? Über den Wolken. Muss die Freiheit wohl grenzenlos sein."
    expected = [
        "What about non-ASCII?",
        "Über den Wolken.",
        "Muss die Freiheit wohl grenzenlos sein."
    ]
    assert split_sentences(text) == expected

def test_split_sentences_lowercase_start():
    # regex requires uppercase after space: (?=[A-Z\u00C0-\u024F])
    text = "This is a sentence. this one starts with lowercase."
    expected = ["This is a sentence. this one starts with lowercase."]
    assert split_sentences(text) == expected

def test_split_sentences_multiple_spaces():
    text = "Sentence one.    Sentence two!"
    expected = ["Sentence one.", "Sentence two!"]
    assert split_sentences(text) == expected

def test_split_sentences_abbreviations():
    # Currently it might split incorrectly if it matches the regex
    text = "Mr. Smith is here. He is a teacher."
    # "Mr. Smith is here." -> split at ". " because "S" is uppercase.
    # Result: ["Mr.", "Smith is here.", "He is a teacher."]
    # But "Mr." is length 3, so it's filtered out.
    # So expected with current implementation:
    expected = ["Smith is here.", "He is a teacher."]
    assert split_sentences(text) == expected

def test_split_sentences_decimal_points():
    text = "The price is 1.50 dollars. It is cheap."
    expected = ["The price is 1.50 dollars.", "It is cheap."]
    assert split_sentences(text) == expected
