## Domain model objects.
from collections import namedtuple


CallSpecification = namedtuple('CallSpecification', ['id', 'dsl_expr_str',
                                                     'evaluation_kwds', 'dependency_values'])

