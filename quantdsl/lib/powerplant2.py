from quantdsl.semantics import Choice, Lift, Market, TimeDelta, Wait, inline


def PowerPlant(start, end, duration_off):
    if (start < end):
        Wait(start,
            Choice(
                ProfitFromRunning(duration_off) + PowerPlant(
                    Tomorrow(start), end, Running()
                ),
                PowerPlant(
                    Tomorrow(start), end, Stopped(duration_off)
                )
            )
        )
    else:
        return 0


@inline
def ProfitFromRunning(duration_off):
    if duration_off > 1:
        return 0.75 * Power() - Gas()
    elif duration_off == 1:
        return 0.90 * Power() - Gas()
    else:
        return 1.00 * Power() - Gas()


@inline
def Power():
    Lift('POWER', 'daily', Market('POWER'))


@inline
def Gas():
    Lift('GAS', 'daily', Market('GAS'))


@inline
def Running():
    return 0


@inline
def Stopped(duration_off):
    return duration_off + 1


@inline
def Tomorrow(today):
    return today + TimeDelta('1d')
