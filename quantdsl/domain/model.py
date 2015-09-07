## Domain model objects.
from collections import namedtuple


class DomainObject(object): pass


CallSpecification = namedtuple('CallSpecification', ['id', 'dsl_expr_str', 'effective_present_time',
                                                     'evaluation_kwds', 'dependency_values'])

# class CallRequirement(DomainObject):
#     def __init__(self, id, dsl_source, effective_present_time, required_call_ids, notify_ids):
#         self.id = id
#         self.dsl_source = dsl_source
#         self.effective_present_time = effective_present_time
#         self.required_call_ids = required_call_ids
#         self.notify_ids = notify_ids
#         # Todo: Validate.
#

class Result(DomainObject):
    def __init__(self, id, return_value):
        self.id = id
        self.value = return_value

