from quantdsl.semantics import Choice, Market, Wait, inline


def GasStorage(start, end, commodity_name, quantity, target, limit, step):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, target),
                Inject(start, end, commodity_name, quantity, limit, step, target, 1),
            ))
        elif quantity >= limit:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, target),
                Inject(start, end, commodity_name, quantity, limit, step, target, -1),
            ))
        else:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, limit, step, target),
                Inject(start, end, commodity_name, quantity, limit, step, target, 1),
                Inject(start, end, commodity_name, quantity, limit, step, target, -1),
            ))
    else:
        if target < 0 or target == quantity:
            0
        else:
            BreachOfContract()


@inline
def BreachOfContract():
    -10000000000000000


@inline
def Continue(start, end, commodity_name, quantity, limit, step, target):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step)


@inline
def Inject(start, end, commodity_name, quantity, limit, step, target, vol):
    Continue(start, end, commodity_name, quantity + vol, limit, step, target) - \
    vol * Market(commodity_name)
