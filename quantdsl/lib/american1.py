from quantdsl.lib.option1 import Option


def American(start, end, strike, underlying, step):
    if start > end:
        0
    else:
        Option(
            start,
            strike,
            underlying,
            American(start + step, end, strike, underlying, step)
        )
