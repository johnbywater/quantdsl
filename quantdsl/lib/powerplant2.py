from quantdsl.semantics import Choice, TimeDelta, Wait, inline, ForwardMarket


def PowerPlant(start, end, temp):
    if (start < end):
        Wait(start, Choice(
            PowerPlant(Tomorrow(start), end, Hot()) + ProfitFromRunning(start, temp),
            PowerPlant(Tomorrow(start), end, Stopped(temp))
        ))
    else:
        return 0

@inline
def Power(start):
    DayAhead(start, 'POWER')

@inline
def Gas(start):
    DayAhead(start, 'GAS')

@inline
def DayAhead(start, name):
    ForwardMarket(Tomorrow(start), name)

@inline
def ProfitFromRunning(start, temp):
    if temp == Cold():
        return 0.3 * Power(start) - Gas(start)
    elif temp == Warm():
        return 0.6 * Power(start) - Gas(start)
    else:
        return Power(start) - Gas(start)

@inline
def Stopped(temp):
    if temp == Hot():
        Warm()
    else:
        Cold()

@inline
def Hot():
    2

@inline
def Warm():
    1

@inline
def Cold():
    0

@inline
def Tomorrow(today):
    today + TimeDelta('1d')
