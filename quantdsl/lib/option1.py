from quantdsl.semantics import Wait, Choice


def Option(date, strike, underlying, alternative):
    return Wait(date, Choice(underlying - strike, alternative))
