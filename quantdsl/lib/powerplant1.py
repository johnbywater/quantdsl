from quantdsl.semantics import Add, Choice, Fixing, Market, Min, Mult, Wait, inline


def PowerPlant(start, end, commodity, cold, step):
    if (start < end):
        Wait(start, Choice(
            Add(
                PowerPlant(start + step, end, commodity, Running(), step),
                ProfitFromRunning(start, commodity, cold)
            ),
            PowerPlant(start + step, end, commodity, Stopped(cold), step),
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
def ProfitFromRunning(start, commodity, cold):
    return Mult((1 - cold / 10), Fixing(start, Burn(commodity)))


@inline
def Burn(commodity):
    return Market(commodity)
