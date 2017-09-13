from quantdsl.semantics import Wait, Choice, inline, Market, Lift


def GasStorage(start, end, commodity_name, quantity, target, limit, step, period):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, 1),
            ))
        elif quantity >= limit:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, -1),
            ))
        else:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, period, target),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, 1),
                Inject(start, end, commodity_name, quantity, limit, step, period, target, -1),
            ))
    else:
        if target < 0 or target == quantity:
            return 0
        else:
            return BreachOfContract()


@inline
def BreachOfContract():
    -10000000000000000

@inline
def Continue(start, end, commodity_name, quantity, limit, step, period, target):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step, period)


@inline
def Inject(start, end, commodity_name, quantity, limit, step, period, target, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step, period, target) - \
    vol * Lift(commodity_name, period, Market(commodity_name))
