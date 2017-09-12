from quantdsl.lib.option1 import Option

def European(date, strike, underlying):
    return Option(date, strike, underlying, 0)
