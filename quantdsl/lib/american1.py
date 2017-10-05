from quantdsl.lib.option1 import Option


def American(start, end, strike, underlying, step):
    if start <= end:
        Option(
            start,
            strike,
            underlying,
            American(start + step, end, strike, underlying, step)
        )
    else:
        0
