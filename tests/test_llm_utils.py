from versarag.llms import ThinkingFilter, strip_thinking


def test_strip_thinking():
    assert strip_thinking("<think>\nhmm\n</think>\n\nAnswer [1]") == "Answer [1]"
    assert strip_thinking("reasoning...</think>Answer") == "Answer"
    assert strip_thinking("Plain answer") == "Plain answer"


def _run(pieces, **kw):
    f = ThinkingFilter(**kw)
    return "".join(f.feed(p) for p in pieces) + f.flush()


def test_thinking_filter_split_tags():
    pieces = ["<th", "ink>\nlet me ", "think</th", "ink>\n\nThe ", "answer is 4", "2."]
    assert _run(pieces) == "The answer is 42."


def test_thinking_filter_no_tags():
    assert _run(["Hello ", "wor", "ld <b>ok</b>"]) == "Hello world <b>ok</b>"


def test_thinking_filter_prefilled():
    assert _run(["reasoning", "</think>", "\nDone"], starts_inside=True) == "Done"
