from quantdsl.semantics import Choice, Market, Wait, inline


def GasStorage(start, end, commodity_name, quantity, target, limit, step, slew):
    if ((start < end) and (limit > 0)):
        if quantity <= 0:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step, slew),
                Inject(start, end, commodity_name, quantity, target, limit, step, slew, slew),
            ))
        elif quantity >= limit:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step, slew),
                Inject(start, end, commodity_name, quantity, target, limit, step, -slew, slew),
            ))
        else:
            return Wait(start, Choice(
                Continue(start, end, commodity_name, quantity, target, limit, step, slew),
                Inject(start, end, commodity_name, quantity, target, limit, step, slew, slew),
                Inject(start, end, commodity_name, quantity, target, limit, step, -slew, slew),
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
def Continue(start, end, commodity_name, quantity, target, limit, step, slew):
    GasStorage(start + step, end, commodity_name, quantity, target, limit, step, slew)


@inline
def Inject(start, end, commodity_name, quantity, target, limit, step, vol, slew):
    Continue(start, end, commodity_name, quantity + vol, target, limit, step, slew) - \
    vol * Market(commodity_name)
