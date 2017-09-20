from quantdsl.semantics import Wait, Choice, inline, Settlement, ForwardMarket


def GasStorage(start, end, commodity_name, quantity, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
            ))
        elif quantity < limit:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
        else:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step),
                Inject(start, end, commodity_name, quantity, limit, step, 1),
                Inject(start, end, commodity_name, quantity, limit, step, -1),
            ))
    else:
        return 0


@inline
def Continue(start, end, commodity_name, quantity, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, limit, step)


@inline
def Continue(start, end, commodity_name, quantity, limit, step):
    GasStorage(start + step, end, commodity_name, quantity, limit, step)


@inline
def Inject(start, end, commodity_name, quantity, limit, step, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step) - \
    Settlement(start, vol * ForwardMarket(commodity_name, start))
