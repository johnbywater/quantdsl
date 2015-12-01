from quantdsl.syntax import DslParser


def dsl_parse(dsl_source, filename='<unknown>', dsl_classes=None):
    """
    Parses DSL module, created according to the given DSL source.
    """
    if dsl_classes is None:
        from quantdsl.semantics import defaultDslClasses
        dsl_classes = defaultDslClasses.copy()

    return DslParser().parse(dsl_source, filename=filename, dsl_classes=dsl_classes)