from quantdsl.semantics import Add, Choice, Fixing, Lift, Market, Min, Mult, Wait, inline


def PowerPlant(start, end, commodity, cold, step, periodisation):
    if (start < end):
        Wait(start, Choice(
            Add(
                PowerPlant(start + step, end, commodity, Running(), step, periodisation),
                ProfitFromRunning(start, commodity, periodisation, cold)
            ),
            PowerPlant(start + step, end, commodity, Stopped(cold), step, periodisation),
        ))
    else:
        return 0


@inline
def Running():
    return 0


@inline
def Stopped(cold):
    return Min(2, cold + 1)


@inline
def ProfitFromRunning(start, commodity, periodisation, cold):
    return Mult((1 - cold / 10), Fixing(start, Burn(commodity, periodisation)))


@inline
def Burn(commodity, periodisation):
    return Lift(commodity, periodisation, Market(commodity))
