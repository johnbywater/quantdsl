## Infrastructure.

from collections import namedtuple

Registry = namedtuple('Registry', ['results', 'calls', 'functions'])

registry = Registry({}, {}, {})

