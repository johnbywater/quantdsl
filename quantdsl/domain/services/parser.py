from quantdsl.syntax import DslParser


def dsl_parse(dsl_source, filename='<unknown>', dsl_classes=None):
    """
    Parses DSL module, created according to the given DSL source.
    """
    if dsl_classes is None:
        dsl_classes = {}

    from quantdsl.semantics import defaultDslClasses
    _dsl_classes = defaultDslClasses.copy()
    _dsl_classes.update(dsl_classes)

    return DslParser(_dsl_classes).parse(dsl_source, filename=filename)